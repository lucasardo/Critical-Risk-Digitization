import streamlit as st
from streamlit_utils import *
from sparql_prompts import *
from dotenv import load_dotenv

################################################################################################

# Credentials
load_dotenv("Retrieval/Azure OpenAI credentials.env")

azure_endpoint = os.environ['GLOBAL_AZURE_ENDPOINT']
openai_api_key = os.environ['GLOBAL_OPENAI_API_KEY']
openai_deployment_name = os.environ['GLOBAL_GPT_DEPLOYMENT_NAME']
openai_api_version = os.environ['GLOBAL_OPENAI_API_VERSION']
embedding_model = os.environ['GLOBAL_EMBEDDING_MODEL']
embedding_deployment_name = os.environ['GLOBAL_EMBEDDING_DEPLOYMENT_NAME']

search_endpoint = os.environ['SEARCH_ENDPOINT']
search_api_key = os.environ['SEARCH_API_KEY']
search_api_version = os.environ['SEARCH_API_VERSION']
search_service_name = os.environ['SEARCH_SERVICE_NAME']
search_url = f"https://{search_service_name}.search.windows.net/"
search_credential = AzureKeyCredential(search_api_key)
search_sem_config = os.environ['SEARCH_SEMANTIC_CONFIG_NAME']

# Models
llm = AzureChatOpenAI(
    deployment_name=openai_deployment_name, 
    openai_api_version=openai_api_version, 
    openai_api_key=openai_api_key, 
    azure_endpoint=azure_endpoint, 
    temperature=0
)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=embedding_deployment_name,
    api_version=openai_api_version,
    api_key=openai_api_key,
    azure_endpoint=azure_endpoint,
)

# Vector stores
doc_index_name: str = "crd-vector-store"
ontology_index_name: str = "crd-ontologies-desc"

# Ontologies
non_inst_ontology_path = os.path.join("Retrieval/ontology/non_inst", "CRA V17.1.ttl")
with open(non_inst_ontology_path, 'r', encoding='utf-8') as file:
        non_inst_ontology = file.read()
        
inst_ontology_path = os.path.join("Retrieval/ontology/inst", "CRA V17.1 MFL & NLE Instantiated Ontology.ttl")
g = Graph()
g.parse(inst_ontology_path)

# Sparql templates

sparql_templates_path = os.path.join("Retrieval/ontology/sparql_templates", "sparql templates.txt")
with open(sparql_templates_path, 'r', encoding='utf-8') as file:
        sparql_templates = file.read()
        
# Chains
_run_manager = CallbackManagerForChainRun.get_noop_manager()
callbacks = _run_manager.get_child()

sparql_generation_chain = sparql_generation_prompt | llm
sparql_fix_chain = sparql_fix_prompt | llm
qa_chain = qa_prompt | llm
no_result_chain = no_result_prompt | llm
combined_chain = combined_prompt | llm

# Classes
sparql_handler = SPARQLQueryHandler(max_sparql_retries=3)
vectorstore_handler = VectorStoreHandler(llm, doc_index_name, ontology_index_name, k=5)
max_sparql_retries = 5  # Max number of attempts at generating a correct SPARQL query
max_query_retries = 3   # Max number of attempts at generating a SPARQL query that returns at least one result
  
################################################################################################
####### SETUP
################################################################################################

st.set_page_config(
    page_title="Graph RAG Agent 💬 WSP",
    page_icon=":robot:",
    layout="wide"
)

# Sidebar
st.sidebar.image("https://download.logo.wine/logo/WSP_Global/WSP_Global-Logo.wine.png", width=100)
st.sidebar.markdown("#")
st.sidebar.write("Try one of the following prompts:")
st.sidebar.write("- List all the Critical Risk Scenarios for Dampier Port Operations")
st.sidebar.write("- List all the Critical Risk Scenarios for Dampier Port Operations along with their descriptions")
st.sidebar.write("- Compare the risks of Richards Bay site and Alma PM site. What insights can you get?")

header = "WSP AI Tool - Critical Risk Agent"
subheader = "Use this AI tool created by WSP to query graph databases using natural language!"

st.markdown(f"<h1 style='color: #F9423A; text-align: center;'>{header}</h1>", unsafe_allow_html=True)
st.markdown(f"<h4 style='color: #F9423A; text-align: center;'>{subheader}</h4>", unsafe_allow_html=True)

st.markdown("#")

################################################################################################

st.markdown("#")
   
if init_prompt := st.chat_input("What would you like to know?"):
    st.chat_message("user").markdown(init_prompt)
    
    input_query = str(init_prompt)
    
    with st.spinner("Generating response..."):
        
        vectorstore_handler.k = 5
        ontology_index_results = vectorstore_handler.perform_graph_search(graph_search_prompt, input_query)

        useful_individuals = []
        for result in ontology_index_results:
            useful_individuals.append(result.page_content)
        
        sparql_generation_chain_result = sparql_generation_chain.invoke(
            {"prompt": input_query, "individuals": str(useful_individuals), "schema": non_inst_ontology, "sparql_queries": sparql_templates}
        )

        raw_sparql = sparql_generation_chain_result.content
        generated_sparql = sparql_handler.get_prepared_sparql_query(_run_manager, sparql_fix_chain, raw_sparql, non_inst_ontology)
        sparql_results = sparql_handler.execute_query(g, generated_sparql)
            
        # If no results, iterate up to a max number of attempts
        query_retries = 0
        while sparql_results == [] and query_retries < max_query_retries:
            
            query_retries += 1

            print(f"### Attempt {query_retries}:", generated_sparql)
            print(f"### No results retrieved by the query, generating a new one...")

            # Use no_result_chain to generate a new query
            no_result_chain_result = no_result_chain.invoke(
                {
                    "generated_sparql": generated_sparql,
                    "prompt": input_query,
                    "schema": non_inst_ontology,
                    "individuals": str(useful_individuals),
                }
            )

            # Get the newly generated SPARQL query and execute it
            generated_sparql = no_result_chain_result.content
            generated_sparql = sparql_handler.get_prepared_sparql_query(_run_manager, sparql_fix_chain, generated_sparql, non_inst_ontology)
            sparql_results = sparql_handler.execute_query(g, generated_sparql)

        # Handle case when no results were retrieved after max iterations
        if sparql_results == []:
            print(f"### No results from graph database after {max_query_retries} attempts for query: {input_query}. Falling back to standard RAG approach.")
        else:
            print("### Successful sparql query:")
            print(generated_sparql)

        vectorstore_handler.k = 5
        # Vector store search
        pdf_index_results = vectorstore_handler.perform_vector_search(pdf_search_prompt, input_query)
        
        # Answer generation
        try:
            combined_chain_result = combined_chain.invoke(
                {
                    "question": input_query, 
                    "sparql_results": sparql_results,  
                    "index_results": pdf_index_results
                }
            )
        except Exception as e:
            print("An error occurred while generating the answer. Falling back to standard RAG.")
            print(e)
            combined_chain_result = combined_chain.invoke(
                {
                    "question": input_query, 
                    "sparql_results": "### The output of the graph query included too many results and exceeded the context window size.",  
                    "index_results": pdf_index_results
                }
            )

        result = combined_chain_result.content
        print("### Generated answer:")
        print(result)

    st.write(result)
    
    # Sources and subgraph
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("Show intermediate steps"):
            st.write("SPARQL Query:")
            st.write(generated_sparql)
            st.write("SPARQL Query Results:")
            st.write(sparql_results)
            st.write("Ontology Vector Store Search:")
            st.write(ontology_index_results)
            st.write("Document Vector Store Search:")
            st.write(pdf_index_results)
            
    with col2: 
        with st.expander("Show subgraph"):
            sparql_construct_query = sparql_construct_prompt + generated_sparql
            construct_output = llm.invoke(sparql_construct_query)
            
            construct_result = construct_output.content
            
            start_index = construct_result.find("PREFIX")
            construct_result = construct_result[start_index:]
            print(construct_result)
            try:
                constructed_graph = g.query(construct_result)

                MAX_LABEL_LENGTH = 15

                # Function to truncate labels
                def truncate_label(label, max_length):
                    if len(label) > max_length:
                        return label[:max_length] + "..."
                    return label

                # Initialize a PyVis network
                net = Network(notebook=True, height="600px", width="600px", directed=True)

                # Customize physics and layout settings
                net.set_options("""
                    var options = {
                    "nodes": {
                        "font": {
                        "size": 16
                        },
                        "scaling": {
                        "min": 10,
                        "max": 30
                        },
                        "labelHighlightBold": true
                    },
                    "edges": {
                        "smooth": true
                    },
                    "physics": {
                        "forceAtlas2Based": {
                        "gravitationalConstant": -50,
                        "centralGravity": 0.01,
                        "springLength": 150,
                        "springConstant": 0.1
                        },
                        "solver": "forceAtlas2Based",
                        "timestep": 0.35,
                        "minVelocity": 0.75
                    },
                    "layout": {
                        "improvedLayout": true
                    },
                    "interaction": {
                        "hover": true,
                        "selectConnectedEdges": false
                    }
                    }
                """)

                # Add nodes and edges to the network with trimmed labels
                for subject, predicate, obj in constructed_graph:
                    # Convert URIs to labels for readability
                    subject_label = subject.split("#")[-1]
                    predicate_label = predicate.split("#")[-1]
                    object_label = obj.split("#")[-1] if isinstance(obj, Namespace) else str(obj)

                    # Truncate labels for nodes
                    subject_display_label = truncate_label(subject_label, MAX_LABEL_LENGTH)
                    object_display_label = truncate_label(object_label, MAX_LABEL_LENGTH)

                    # Add nodes with truncated labels and tooltips showing full text
                    net.add_node(subject_label, label=subject_display_label, title=subject_label, size=20)
                    net.add_node(object_label, label=object_display_label, title=object_label, size=20)
                    net.add_edge(subject_label, object_label, label=predicate_label)

                # Save the network to an HTML file
                net.show("subgraph.html")
                path_to_html = "subgraph.html"

                # Embed the HTML in Streamlit
                with open(path_to_html, 'r') as f:
                    html_data = f.read()

                st.components.v1.html(html_data, height=600, width=600)
            
            except:
                st.write("Unable to generate subgraph for this query.")