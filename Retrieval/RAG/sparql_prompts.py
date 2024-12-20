from langchain_community.chains.graph_qa.prompts import (
    GRAPHDB_QA_PROMPT,
    GRAPHDB_SPARQL_FIX_PROMPT,
    GRAPHDB_SPARQL_GENERATION_PROMPT,
)
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.base import BasePromptTemplate
from pydantic import Field
from langchain_core.prompts import ChatPromptTemplate

#### Prompts

SPARQL_SELECT_TEMPLATE = """
Task: Generate a SPARQL SELECT statement for querying a graph database based on a natural language question.

For instance, a question could be: "List all the Critical Risk Scenarios that impact 'Dampier Port'".

The following sparql query in backticks would be suitable:
```
PREFIX cro: <http://WSP.org/ontology/cro#>
SELECT ?criticalRiskScenario
WHERE {{
    ?operation cro:hasName "Dampier" .
    ?criticalRiskScenario cro:impactsOperation ?operation .
}}
```
Instructions:
Use only the node types and properties provided in the schema.
Do not use any node types and properties that are not explicitly provided.
Do not wrap the query in backticks.
Include all necessary prefixes.
Schema:
{schema}
The 'Operation' class names are "Dampier", "Cape Lambert Operations" and "Oyu Tolgoi Copper Mine".
Note: Be as concise as possible.
Do not include any explanations or apologies in your responses.
Do not respond to any questions that ask for anything else than for you to construct a SPARQL query.
Do not include any text except the SPARQL query generated.

Useful individuals for classes and properties that can help formulating the query:
{individuals}

Examples of pre-built SPARQL queries that you can leverage to generate the new query:
{sparql_queries}

The natural language question is:
{prompt}
"""
SPARQL_GENERATION_SELECT_PROMPT = PromptTemplate(
    input_variables=["schema", "individuals", "sparql_queries", "prompt"], 
    template=SPARQL_SELECT_TEMPLATE
)
    
SPARQL_NO_RESULT_TEMPLATE = """
This following SPARQL query delimited by triple backticks
```
{generated_sparql}
```
is valid, but it didn't return any results from the graph.
Please try a slightly different SPARQL query.
This is the question in natural language that should be answered by the query, delimited by triple backticks:
```
{prompt}
```
Do not include any explanations or apologies in your responses.
Do not wrap the query in backticks.
Do not include any text except the SPARQL query generated.
The ontology schema delimited by triple backticks in Owl format is:
```
{schema}
```
Useful individuals for classes and properties that can help formulating the query:
{individuals}
"""

SPARQL_NO_RESULT_PROMPT = PromptTemplate(
    input_variables=["generated_sparql", "prompt", "schema", "individuals"],
    template=SPARQL_NO_RESULT_TEMPLATE,
)

COMBINED_QA_TEMPLATE = """
Generate a natural language response using the following sources as context: 
    
1. The results of a SPARQL query (primary source)
2. The results of a search in a vector store containing document chunks (secondary source)

Both sources are provided to you as context.
Use the primary source to generate the response, and use the secondary source to provide additional context if needed. In the response, you should not explicitly say if the information comes from the primary or secondary source.
The information provided is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Don't use internal knowledge to answer the question, if no information is available juts say that there is no information available.

### User's question: {question}

### Sparql Results: {sparql_results}

### Vector store results: {index_results}

### Examples of how to format the answer to the user queries

The following is an example of how to format the answer to the user queries, do not use the content in these examples to generate responses for the user queries:
  
--> Beginning of example
  
Example 1:
  
Question:
What are the Critical Risk Scenarios for this site? 

Answer:
These are the Critical Risk Scenarios along with their respective incident descriptions:

1. **DPO-01**: A 1 in 100-year Category 5 tropical cyclone directly impacts the Karratha and Dampier townships and the Dampier port operations causing extensive damage in the region and storm surge and flooding. 

2. **DPO-04**: One of the two Parker Point shiploaders is lost. The loss could be caused by extreme weather conditions (high wind and/or wave action), structural failure due to fatigue (of either the wharf or shiploader), exceeding design loads, high impact collision, or the loss of stability caused by a vessel rising on an incoming tide impacting a shiploader boom that is unable to move clear or some combination of these.

3. **DPO-05**: The dredged portion of the common departure channel for both EII and PPt (between "Middle Ground" and "Fairway" markers) becomes blocked due to a grounded ship. This impacts loaded vessels departing from both PPt and EII. The most likely causes could be due to a collision, on-board fire, or grounding. The sunken ship prevents the passage of fully loaded bulk carriers.
    
<-- End of example

"""

COMBINED_QA_PROMPT = PromptTemplate(
    input_variables=["sparql_results", "index_results", "prompt"], template=COMBINED_QA_TEMPLATE
)

pdf_search_prompt = """
### Instructions
Your task is to rephrase queries coming from users into one sentence queries. 
The output will be used as a query to search in a database for documents relevant to the question. 
The output should be only a text string. 

User's query: 
"""

graph_search_prompt = """
### Instructions
Your task is to rephrase queries coming from users into one sentence queries. 
The generated output will be used as a query to search in a database that contains nodes extracted from a RDF knwoledge graph.
The rephrased query should highlight the key classes and properties that could help identify the most relevant nodes in the knowledge graph.
Do not change the meaning or the intent of the user query.
The output should be only a text string. 

### Example:
- User's query: 
"What Risk Management Sub Elements showed the greatest increase for the current year compared to the previous survey at Cape Lambert Operations? What insights can you gather from these observations? "
- Rephrased query:
"Risk Management Sub Elements with the greatest increase this year compared to the previous survey at Cape Lambert Operations."

User's query: 
"""