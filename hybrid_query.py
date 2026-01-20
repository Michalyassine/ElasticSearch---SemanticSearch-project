import os
import sys
from typing import List, Dict, Any

# LangChain & RAG Components
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Elasticsearch Client
from elasticsearch import Elasticsearch

# Config
import config

# ==========================================
# 1. INITIALIZATION
# ==========================================

def setup_llm():
    """Setup LLM using OpenRouter API."""
    if not config.OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not found.")
        return None
    try:
        llm = ChatOpenAI(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=config.OPENROUTER_API_KEY,
            model_name=config.LLM_MODEL_NAME,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS
        )
        return llm
    except Exception as e:
        print(f"Error connecting to LLM: {e}")
        return None

def setup_vector_store(embeddings):
    """Initialize ChromaDB (Semantic)."""
    try:
        vectordb = Chroma(
            persist_directory=config.PERSIST_DIRECTORY,
            embedding_function=embeddings,
            collection_name=config.COLLECTION_NAME
        )
        return vectordb
    except Exception as e:
        print(f"Error connecting to Vector Store: {e}")
        return None

def setup_elasticsearch():
    """Initialize Elasticsearch Client (Syntaxic)."""
    try:
        # Connect to the local Elasticsearch container
        es = Elasticsearch("http://localhost:9200")
        if es.ping():
            print("Connected to Elasticsearch.")
            return es
        else:
            print("Could not ping Elasticsearch.")
            return None
    except Exception as e:
        print(f"Error connecting to Elasticsearch: {e}")
        return None

# ==========================================
# 2. SEARCH ENGINES
# ==========================================

def search_elasticsearch(es_client, query_text, index_name="mozilla-log-*", size=10):
    """
    Performs a Syntaxic (Keyword) search in Elasticsearch.
    Returns a list of LangChain Documents.
    """
    print(f"\n[Elasticsearch] Searching for: '{query_text}'...")
    
    # 1. Define the Query
    # We use a simple_query_string which allows for loose matching
    # and boolean operators (AND, OR) naturally in the search bar.
    query_body = {
        "size": size,
        "query": {
            "simple_query_string": {
                "query": query_text,
                "fields": ["event_message^2", "raw_line", "log_level"], 
                # ^2 boosts the relevance of the event_message field
                "default_operator": "or"
            }
        }
    }

    try:
        # 2. Execute Search
        response = es_client.search(index=index_name, body=query_body)
        hits = response['hits']['hits']
        
        docs = []
        for hit in hits:
            source = hit['_source']
            score = hit['_score']
            
            # Extract text content. 
            # Prefer 'event_message' (parsed) or fallback to 'raw_line'
            content = source.get("event_message") or source.get("raw_line") or str(source)
            
            # Create LangChain Document
            doc = Document(
                page_content=content,
                metadata={
                    "source": "Elasticsearch",
                    "score": score,
                    "index": hit['_index'],
                    "timestamp": source.get("@timestamp", "N/A")
                }
            )
            docs.append(doc)
            
        print(f"  -> Found {len(docs)} documents.")
        return docs
        
    except Exception as e:
        print(f"  -> Error searching Elasticsearch: {e}")
        return []

def search_vector_store(vectordb, query_text, k=10):
    """
    Performs a Semantic Search in ChromaDB.
    Returns a list of LangChain Documents.
    """
    print(f"\n[Vector DB] Searching for: '{query_text}'...")
    try:
        # Perform similarity search
        docs = vectordb.similarity_search(query_text, k=k)
        print(f"  -> Found {len(docs)} documents.")
        return docs
    except Exception as e:
        print(f"  -> Error searching Vector DB: {e}")
        return []

# ==========================================
# 3. HYBRID QUERY LOGIC
# ==========================================

def run_hybrid_query(es_client, vectordb, llm, user_query):
    """
    Executes the hybrid workflow:
    1. Search ES (Syntaxic)
    2. Search Vector DB (Semantic)
    3. Merge results
    4. Generate Answer
    """
    
    # 1. Parallel Search
    es_docs = search_elasticsearch(es_client, user_query, size=10)
    semantic_docs = search_vector_store(vectordb, user_query, k=10)
    
    # 2. Merge Strategy
    # Simple concatenation for this example.
    # Ideally, you would rank them, but for the LLM "Stuff" method,
    # feeding it diverse context is usually enough.
    combined_docs = es_docs + semantic_docs
    
    # Optional: Deduplication based on content to avoid sending the exact same line twice
    unique_docs = []
    seen_content = set()
    for doc in combined_docs:
        content_hash = hash(doc.page_content)
        if content_hash not in seen_content:
            unique_docs.append(doc)
            seen_content.add(content_hash)
            
    print(f"\n[Hybrid Engine] Merged results. Total unique context lines: {len(unique_docs)}")

    # 3. Generation (RAG)
    if not unique_docs:
        print("No documents found in either system.")
        return

    # Prepare the prompt
    prompt_template = """
    You are an expert system administrator analyzing Firefox build logs.
    I have provided you with context retrieved from two systems: 
    1. Keyword-based search (Syntaxic)
    2. Vector-based search (Semantic)

    Use the following pieces of retrieved context to answer the user's question.
    If the answer is not in the context, say that you don't know.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # We use a custom chain here because we are injecting our own list of documents
    # rather than using a standard retriever automatically.
    context_text = "\n".join([doc.page_content for doc in unique_docs])
    
    print("\n[LLM] Generating response...")
    
    if not llm:
        print("LLM not configured. Printing raw context instead:")
        print(context_text[:500] + "...")
        return

    try:
        # Format the prompt manually
        final_prompt = PROMPT.format(context=context_text, question=user_query)
        
        # Invoke LLM
        response = llm.invoke(final_prompt)
        
        print("\n" + "="*60)
        print(f"FINAL ANSWER:\n{response.content}")
        print("="*60)
        
    except Exception as e:
        print(f"Error generating response: {e}")

# ==========================================
# 4. MAIN LOOP
# ==========================================

def main():
    print("Initializing Hybrid Query System...")
    
    # 1. Load Embeddings (Required for Vector DB)
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': config.EMBEDDING_DEVICE},
    )
    
    # 2. Initialize Services
    vectordb = setup_vector_store(embeddings)
    es_client = setup_elasticsearch()
    llm = setup_llm()

    if not vectordb or not es_client:
        print("Failed to initialize search engines. Exiting.")
        return

    print("\n" + "="*60)
    print("SYSTEM READY. Ask questions about your logs.")
    print("Type 'exit' to quit.")
    print("="*60)

    # 3. Query Loop
    while True:
        try:
            user_query = input("\nUser Query > ")
            
            if user_query.lower() == 'exit':
                break
            
            if not user_query.strip():
                continue

            # Run Hybrid Query
            run_hybrid_query(es_client, vectordb, llm, user_query)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()