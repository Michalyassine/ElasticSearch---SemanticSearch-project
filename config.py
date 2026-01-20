# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# --- Kafka Configuration ---
KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'localhost:9094')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'log-topic')

# --- Vector DB Configuration ---
PERSIST_DIRECTORY = './chroma_db_storage'
COLLECTION_NAME = "log_analysis"

# --- Batch Update Configuration ---
BATCH_UPDATE_INTERVAL_SEC = 300  # 5 minutes

# --- Embedding Model (Local) ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu" 

# --- LLM Configuration (OpenRouter) ---
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
LLM_MODEL_NAME = "openai/gpt-3.5-turbo" 
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 512

# --- RAG Retrieval Settings ---
RETRIEVAL_TOP_K = 100

# --- Vector DB Safety Configuration ---
# Max documents to send to ChromaDB at once (Default limit is often ~5461)
CHROMA_BATCH_SIZE = 4000 