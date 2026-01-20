import os
from kafka import KafkaProducer
import json
import time
import random

# --- Configuration ---
LOGS_ROOT_DIR = 'D:/01_ENSIAS/05_S5/IR/project/Data2'
KAFKA_BROKER = 'localhost:9094'  
KAFKA_TOPIC = 'log-topic'

def create_kafka_producer():
    """Initializes and returns a Kafka Producer configured for JSON serialization."""
    print(f"Attempting to connect to Kafka broker at {KAFKA_BROKER}...")
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            # Use JSON serializer to send structured data
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            retries=5,
            retry_backoff_ms=500
        )
        print("Kafka Producer connected successfully.")
        return producer
    except Exception as e:
        print(f"Error connecting to Kafka: {e}")
        return None

def process_log_files(producer):
    """
    Reads log files from the directory and sends them to Kafka.
    Includes a small sleep to simulate real-time streaming.
    """
    if not os.path.isdir(LOGS_ROOT_DIR):
        print(f"Error: Directory '{LOGS_ROOT_DIR}' not found.")
        return

    total_lines = 0
    total_files = 0
    
    print(f"Starting to stream logs from {LOGS_ROOT_DIR}...")
    
    for root, dirs, files in os.walk(LOGS_ROOT_DIR):
        for filename in files:
            # Filter for text files (adjust extension if your logs are .log, etc.)
            if filename.endswith(".txt") or filename.endswith(".log"):
                file_path = os.path.join(root, filename)
                print(f"\nProcessing file: {file_path}")
                file_lines = 0
                total_files += 1
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            # Skip empty lines
                            if not line.strip():
                                continue

                            # SIMPLE PAYLOAD: Keep text as is
                            payload = {
                                "content": line.strip(),  # The raw log line
                                "source_file": filename    # Optional: keep track of file name
                            }
                            
                            # Send to Kafka
                            producer.send(KAFKA_TOPIC, value=payload)
                            total_lines += 1
                            file_lines += 1
                            
                            # Simulate real-time streaming with a tiny delay
                            # Increase this number (e.g., 0.01) if you want to slow it down
                            time.sleep(0.001) 
                            
                    print(f"  -> Streamed {file_lines} lines from {filename}")
                        
                except Exception as e:
                    print(f"  Failed to process {filename}: {e}")

    # Ensure all data is sent before exiting
    producer.flush() 
    print(f"\n--- Streaming Complete ---")
    print(f"Total lines sent to topic '{KAFKA_TOPIC}': {total_lines}")

if __name__ == "__main__":
    producer = create_kafka_producer()
    if producer:
        process_log_files(producer)