#### Imports

import os    
from sparql_prompts import *

from azure.core.credentials import AzureKeyCredential
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings    
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain_core.callbacks.manager import CallbackManager, CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.base import BasePromptTemplate
from pydantic import Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.chains.graph_qa.prompts import (
    GRAPHDB_QA_PROMPT,
    GRAPHDB_SPARQL_FIX_PROMPT,
    GRAPHDB_SPARQL_GENERATION_PROMPT,
)
from langchain_community.graphs import OntotextGraphDBGraph
from IPython.display import Image, display

from langsmith import Client
from langsmith import traceable

from langchain_community.graphs import OntotextGraphDBGraph
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import rdflib
    
from rdflib import Graph, Namespace
from rdflib.plugins.sparql import prepareQuery

import textwrap
import pandas as pd
from pyvis.network import Network
import requests
from typing import List, OrderedDict
import json

from dotenv import load_dotenv
load_dotenv("Retrieval/Azure OpenAI credentials.env")

# Credentials
search_endpoint = os.environ['SEARCH_ENDPOINT']
search_api_key = os.environ['SEARCH_API_KEY']
search_api_version = os.environ['SEARCH_API_VERSION']
search_service_name = os.environ['SEARCH_SERVICE_NAME']
search_url = f"https://{search_service_name}.search.windows.net/"
search_credential = AzureKeyCredential(search_api_key)
search_sem_config = os.environ['SEARCH_SEMANTIC_CONFIG_NAME']

# Prompts

sparql_generation_prompt: BasePromptTemplate = SPARQL_GENERATION_SELECT_PROMPT
sparql_fix_prompt: BasePromptTemplate = GRAPHDB_SPARQL_FIX_PROMPT
sparql_fix_prompt: BasePromptTemplate = GRAPHDB_SPARQL_FIX_PROMPT
qa_prompt: BasePromptTemplate = GRAPHDB_QA_PROMPT
no_result_prompt: BasePromptTemplate = SPARQL_NO_RESULT_PROMPT
combined_prompt: BasePromptTemplate = COMBINED_QA_PROMPT

# Classes

class SPARQLQueryHandler:
    def __init__(self, max_sparql_retries: int = 3):
        self.max_sparql_retries = max_sparql_retries

    def log_prepared_sparql_query(self, _run_manager, generated_query: str) -> None:
        _run_manager.on_text("Generated SPARQL:", end="\n", verbose=True)
        _run_manager.on_text(generated_query, color="green", end="\n", verbose=True)

    def log_invalid_sparql_query(self, _run_manager, generated_query: str, error_message: str) -> None:
        _run_manager.on_text("Invalid SPARQL query: ", end="\n", verbose=True)
        _run_manager.on_text(generated_query, color="red", end="\n", verbose=True)
        _run_manager.on_text("SPARQL Query Parse Error: ", end="\n", verbose=True)
        _run_manager.on_text(error_message, color="red", end="\n\n", verbose=True)

    def prepare_sparql_query(self, _run_manager, generated_sparql: str) -> str:
        prepareQuery(generated_sparql)
        self.log_prepared_sparql_query(_run_manager, generated_sparql)
        return generated_sparql

    def get_prepared_sparql_query(self, _run_manager, sparql_fix_chain, generated_sparql: str, ontology_schema: str) -> str:
        try:
            return self.prepare_sparql_query(_run_manager, generated_sparql)
        except Exception as e:
            retries = 0
            error_message = str(e)
            self.log_invalid_sparql_query(_run_manager, generated_sparql, error_message)
            print("Error message: ", error_message)

            while retries < self.max_sparql_retries:
                try:
                    sparql_fix_chain_result = sparql_fix_chain.invoke(
                        {
                            "error_message": error_message,
                            "generated_sparql": generated_sparql,
                            "schema": ontology_schema,
                        }
                    )
                    generated_sparql = sparql_fix_chain_result.content
                    return self.prepare_sparql_query(_run_manager, generated_sparql)
                except Exception as e:
                    retries += 1
                    parse_exception = str(e)
                    print("Error message (parse_exception): ", parse_exception)
                    self.log_invalid_sparql_query(_run_manager, generated_sparql, parse_exception)

            print("The generated SPARQL query is invalid.")
            return None

    def execute_query(self, g, query: str) -> List[rdflib.query.ResultRow]:
        try:
            rdf_results = g.query(query)

            results_list = []
            for row in rdf_results:
                results_list.append(row)

            return results_list
        except Exception:
            print("Failed to execute the generated SPARQL query.")
            return []

class VectorStoreHandler:
    def __init__(self, llm, doc_index_name, ontology_index_name, k: int = 5):
        self.llm = llm
        self.k = k
        self.doc_index_name = doc_index_name
        self.ontology_index_name = ontology_index_name
        self.headers = {'Content-Type': 'application/json',
            'api-key': search_api_key}
        self.params = {'api-version': search_api_version}
    
    def perform_vector_search(self, system_prompt: str, input_query: str):
        raw_query = str(system_prompt + input_query)    
        search_query = self.llm.invoke(raw_query)
        
        search_query = str(search_query.content)
        print("### Querying pdf document database: ", search_query)
                    
        # Documents Vector Store        
        search_payload = {
            "search": search_query,
            "queryType": "semantic",
            "vectorQueries": [{"text": search_query, "fields": "embedding", "kind": "text"}],
            "semanticConfiguration": search_sem_config,
            "captions": "extractive",
            "answers": "extractive|count-3",
            "queryLanguage": "en-us",
            "count": "true",
            "top": self.k
        }
        
        resp = requests.post(f"{search_endpoint}/indexes/{self.doc_index_name}/docs/search",
                            data=json.dumps(search_payload), headers=self.headers, params=self.params)
        search_results = resp.json()

        content = dict()
        ordered_content = OrderedDict()

        for result in search_results['value']:
            if result['@search.rerankerScore'] > 1: #Change treshold if needed
                content[result['id']] = {
                    "doc_path": result['doc_path'],
                    "chunk": result['chunk'],
                    "score": result['@search.rerankerScore']
                }

        topk = self.k
        
        count = 0  # To keep track of the number of results added
        for id in sorted(content, key=lambda x: content[x]["score"], reverse=True):
            ordered_content[id] = content[id]
            count += 1
            if count >= topk:  # Stop after adding topK results
                break
            
        index_results = []
        for key, value in ordered_content.items():
            try:
                index_results.append(Document(page_content=value["chunk"], metadata={
                    "doc_path": value['doc_path'], "score": value["score"]}))

            except:
                print("An exception occurred")
    
        return index_results
        
    def perform_graph_search(self, system_prompt: str, input_query: str):
        
        raw_query = str(system_prompt + input_query)    
        search_query = self.llm.invoke(raw_query)
        
        search_query = str(search_query.content)
        print("### Querying ontology database: ", search_query)
        
        # Ontology Vector Store

        search_payload = {
            "search": search_query,
            "queryType": "semantic",
            "vectorQueries": [{"text": search_query, "fields": "embedding", "kind": "text"}],
            "semanticConfiguration": search_sem_config,
            "captions": "extractive",
            "answers": "extractive|count-3",
            "queryLanguage": "en-us",
            "filter": "doc_path eq 'CRA V16.2 MFL Instantiated Ontology.ttl'",
            "count": "true",
            "top": self.k
        }
        
        resp = requests.post(f"{search_endpoint}/indexes/{self.ontology_index_name}/docs/search",
                            data=json.dumps(search_payload), headers=self.headers, params=self.params)
        search_results = resp.json()

        content = dict()
        ordered_content = OrderedDict()

        for result in search_results['value']:
            if result['@search.rerankerScore'] > 1: #Change treshold if needed
                content[result['id']] = {
                    "doc_path": result['doc_path'],
                    "individual": result['individual'],
                    "score": result['@search.rerankerScore']
                }

        topk = self.k
        
        count = 0  # To keep track of the number of results added
        for id in sorted(content, key=lambda x: content[x]["score"], reverse=True):
            ordered_content[id] = content[id]
            count += 1
            if count >= topk:  # Stop after adding topK results
                break
            
        index_results = []
        for key, value in ordered_content.items():
            try:
                index_results.append(Document(page_content=value["individual"], metadata={
                    "doc_path": value['doc_path'], "score": value["score"]}))

            except:
                print("An exception occurred")
    
        return index_results