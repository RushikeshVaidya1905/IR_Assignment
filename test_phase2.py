from src.parsing.reuters_parser import ReutersParser
from src.parsing.preprocessor import TextPreprocessor
from src.indexing.spimi import SPIMIIndexer
import time

def main():
    raw_dir = "data/raw"
    index_dir = "index_data/shard_0"  # We will test on a single shard directory first
    
    parser = ReutersParser(raw_dir)
    preprocessor = TextPreprocessor(use_stemming=True, remove_stopwords=True)
    
    # We set a tiny memory limit (e.g., 25 MB) to force multiple block flushes 
    # during testing, so you can see it working without needing massive datasets.
    spimi = SPIMIIndexer(output_dir=index_dir, memory_limit_mb=25.0)

    print("--- Starting Phase 2: SPIMI Block Generation ---")
    start_time = time.time()
    
    doc_count = 0
    # Process only the first 500 documents for this quick test
    for doc in parser.parse_all():
        doc_count += 1
        
        # Process Title
        title_tokens = preprocessor.process_with_positions(doc.title)
        for token, pos in title_tokens:
            spimi.add_token(token, doc.doc_id, pos)
            
        # Process Body (offsetting positions by title length to keep them sequential)
        title_len = len(doc.title.split()) if doc.title else 0
        body_tokens = preprocessor.process_with_positions(doc.body)
        for token, pos in body_tokens:
            spimi.add_token(token, doc.doc_id, pos + title_len)

        # Check memory budget every 50 documents to avoid overhead
        if doc_count % 50 == 0:
            if spimi.is_memory_full():
                print(f"Memory threshold reached ({spimi.get_current_memory_usage():.2f} MB). Flushing...")
                spimi.flush_to_disk()
                
        if doc_count == 500:
            break

    # Flush any remaining data in memory at the end
    spimi.flush_to_disk()
    
    print(f"\nProcessed {doc_count} documents in {time.time() - start_time:.2f} seconds.")
    print(f"Check the '{index_dir}' folder to see the generated JSON blocks.")

if __name__ == "__main__":
    main()