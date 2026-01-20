import os
import time
import json
from typing import List

from kafka import KafkaConsumer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
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
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS
        )
        print("LLM connected successfully.")
        return llm
    except Exception as e:
        print(f"Error connecting to LLM: {e}")
        return None

# ==========================================
# 2. PARALLEL INGESTION LOGIC
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
            group_id='hybrid-consumer-group', # New consumer group
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        print("Kafka Consumer connected.")
        return consumer
    except Exception as e:
        print(f"Error connecting to Kafka: {e}")
        raise

def process_kafka_messages(consumer, timeout=5.0):
    """
    Reads messages from Kafka.
    Note: Logstash is consuming in parallel to populate Elasticsearch.
    We are consuming here to populate the Vector DB.
    """
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
                    # Create Document for Vector DB
                    doc = Document(page_content=log_content, metadata={"source": source_file})
                    batch_docs.append(doc)
                    
    return batch_docs

def update_vector_store(vectordb, documents: List[Document]):
    """
    Updates Vector Store in batches.
    """
    if not documents:
        print("No new logs to process for Vector DB.")
        return

    total_docs = len(documents)
    print(f"Updating Vector Store with {total_docs} new log lines...")
    print(f"(Note: Logstash is handling Elasticsearch indexing in parallel)")
    
    for i in range(0, total_docs, config.CHROMA_BATCH_SIZE):
        batch = documents[i : i + config.CHROMA_BATCH_SIZE]
        
        try:
            vectordb.add_documents(batch)
            vectordb.persist()
            print(f"  > Processed Vector DB batch {i // config.CHROMA_BATCH_SIZE + 1} ({len(batch)} docs)")
        except Exception as e:
            print(f"  > Error processing Vector DB batch starting at index {i}: {e}")

# ==========================================
# 3. MAIN INGESTION LOOP
# ==========================================

def main():
    # Setup Components
    embeddings = setup_embeddings()
    vectordb = setup_vector_store(embeddings)
    consumer = get_kafka_consumer()
    # We don't need LLM for ingestion, but good to check config
    llm = setup_llm()

    print("\n" + "="*50)
    print("HYBRID INGESTION SYSTEM STARTED")
    print("This script will:")
    print("1. Read logs from Kafka")
    print("2. Index them into ChromaDB (Semantic)")
    print("3. (Logstash handles Elasticsearch (Syntaxic) automatically)")
    print("="*50)

    # Run for a fixed duration or until manual stop
    # For this test, let's just run for 2 cycles then stop so we can verify
    cycle_count = 0
    max_cycles = 3 

    try:
        while cycle_count < max_cycles:
            cycle_count += 1
            print(f"\n--- Cycle {cycle_count}/{max_cycles} ---")
            
            # 1. Fetch from Kafka
            # Note: We use a longer timeout here to grab data sent by producer
            new_docs = process_kafka_messages(consumer, timeout=10.0)
            
            # 2. Update Vector DB
            update_vector_store(vectordb, new_docs)
            
            if new_docs:
                print(f"Sleeping for {config.BATCH_UPDATE_INTERVAL_SEC} seconds...")
                time.sleep(config.BATCH_UPDATE_INTERVAL_SEC)
            else:
                print("No data found. Exiting.")
                break

    except KeyboardInterrupt:
        print("\nStopping ingestion...")
    finally:
        if consumer:
            consumer.close()
        print("Ingestion complete.")

if __name__ == "__main__":
    main()