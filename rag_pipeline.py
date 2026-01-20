import os
import time
import json
from typing import List

from kafka import KafkaConsumer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Import all configuration variables
import config

# ==========================================
# 1. INITIALIZATION
# ==========================================

def setup_embeddings():
    """Initialize the lightweight embedding model."""
    print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': config.EMBEDDING_DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("Embedding model loaded.")
        return embeddings
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        raise

def setup_vector_store(embeddings):
    """Initialize ChromaDB."""
    print(f"Connecting to Vector Store at {config.PERSIST_DIRECTORY}...")
    try:
        vectordb = Chroma(
            persist_directory=config.PERSIST_DIRECTORY,
            embedding_function=embeddings,
            collection_name=config.COLLECTION_NAME
        )
        print("Vector Store ready.")
        return vectordb
    except Exception as e:
        print(f"Error connecting to Vector Store: {e}")
        raise

def setup_llm():
    """Setup LLM using OpenRouter API."""
    if not config.OPENROUTER_API_KEY:
        print("WARNING: OPENROUTER_API_KEY not found.")
        return None

    print(f"Connecting to OpenRouter LLM: {config.LLM_MODEL_NAME}...")
    try:
        llm = ChatOpenAI(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=config.OPENROUTER_API_KEY,
            model_name=config.LLM_MODEL_NAME,
            temperature=config.LLM_TEMPERATURE
            #,max_tokens=config.LLM_MAX_TOKENS
        )
        print("LLM connected successfully.")
        return llm
    except Exception as e:
        print(f"Error connecting to LLM: {e}")
        return None

# ==========================================
# 2. KAFKA CONSUMER & BATCH PROCESSING
# ==========================================

def get_kafka_consumer():
    """Initialize the Kafka Consumer."""
    print(f"Connecting to Kafka at {config.KAFKA_BROKER}...")
    try:
        consumer = KafkaConsumer(
            config.KAFKA_TOPIC,
            bootstrap_servers=[config.KAFKA_BROKER],
            auto_offset_reset='latest', 
            enable_auto_commit=True,
            group_id='rag-consumer-group',
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        print("Kafka Consumer connected.")
        return consumer
    except Exception as e:
        print(f"Error connecting to Kafka: {e}")
        raise

def process_kafka_messages(consumer, timeout=1.0):
    """Reads messages from Kafka and accumulates them."""
    print("--- Listening for new logs ---")
    batch_docs = []
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        msg_pack = consumer.poll(timeout_ms=1000)
        for tp, messages in msg_pack.items():
            for message in messages:
                log_content = message.value.get("content", "")
                source_file = message.value.get("source_file", "unknown")
                
                if log_content:
                    doc = Document(page_content=log_content, metadata={"source": source_file})
                    batch_docs.append(doc)
                    
    return batch_docs

def update_vector_store(vectordb, documents: List[Document]):
    """
    Adds documents to the Vector Store in chunks to avoid exceeding batch limits.
    """
    if not documents:
        return

    total_docs = len(documents)
    print(f"Updating Vector Store with {total_docs} new log lines...")
    
    # Loop through documents in steps defined by CHROMA_BATCH_SIZE
    for i in range(0, total_docs, config.CHROMA_BATCH_SIZE):
        batch = documents[i : i + config.CHROMA_BATCH_SIZE]
        
        try:
            vectordb.add_documents(batch)
            # We persist after every chunk to ensure data is safe
            vectordb.persist()
            print(f"  > Processed batch {i // config.CHROMA_BATCH_SIZE + 1} ({len(batch)} docs)")
        except Exception as e:
            print(f"  > Error processing batch starting at index {i}: {e}")

# ==========================================
# 3. RAG QUERY LOGIC
# ==========================================

def query_rag_system(vectordb, llm, query_text):
    """Performs retrieval using High K and generates answer."""
    print(f"Retrieving Top {config.RETRIEVAL_TOP_K} relevant docs...")
    
    retriever = vectordb.as_retriever(
        search_kwargs={"k": config.RETRIEVAL_TOP_K}
    )
    
    prompt_template = """
    You are a system administrator analyzing server logs.
    Use the retrieved log lines to answer the user's question.
    If the logs do not contain the answer, state that clearly.
    Keep the answer concise and technical.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    if not llm:
        print("LLM not available. Showing retrieved context instead.")
        relevant_docs = retriever.get_relevant_documents(query_text)
        for d in relevant_docs:
            print(f" - {d.page_content}")
        return

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    try:
        response = qa_chain({"query": query_text})
        print("\n=== Retrieved Context ===")
        for doc in response['source_documents']:
            print(f" - {doc.page_content}")
        print("\n=== LLM Response ===")
        print(response['result'])
        print("====================")
    except Exception as e:
        print(f"Error during generation: {e}")

# ==========================================
# 4. MAIN LOOP
# ==========================================

def main():
    embeddings = setup_embeddings()
    vectordb = setup_vector_store(embeddings)
    consumer = get_kafka_consumer()
    llm = setup_llm()

    print("\n" + "="*50)
    print("SYSTEM READY.")
    print(f"Update Interval: {config.BATCH_UPDATE_INTERVAL_SEC}s")
    print("Type 'exit' to quit.")
    print("="*50)

    last_update_time = time.time()
    
    try:
        while True:
            current_time = time.time()
            
            # --- A. UPDATE CYCLE ---
            if current_time - last_update_time >= config.BATCH_UPDATE_INTERVAL_SEC:
                # Note: Adjusted timeout slightly to catch more logs per cycle if needed
                new_docs = process_kafka_messages(consumer, timeout=5.0)
                update_vector_store(vectordb, new_docs)
                
                last_update_time = current_time
                print(f"Next update in {config.BATCH_UPDATE_INTERVAL_SEC} seconds.")

            # --- B. QUERY INTERFACE ---
            user_query = input("\nEnter question (or wait): ")
            
            if user_query.lower() == 'exit':
                break
            
            if user_query.strip():
                query_rag_system(vectordb, llm, user_query)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if consumer:
            consumer.close()

if __name__ == "__main__":
    main()