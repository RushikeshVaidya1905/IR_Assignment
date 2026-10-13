import os
import glob
import time
from multiprocessing import Pool
from src.parsing.reuters_parser import ReutersParser
from src.parsing.preprocessor import TextPreprocessor
from src.indexing.spimi import SPIMIIndexer

def build_shard(shard_info):
    """Worker function to build a complete SPIMI index for a subset of files."""
    shard_id, file_paths = shard_info
    index_dir = f"index_data/shard_{shard_id}"
    
    preprocessor = TextPreprocessor(use_stemming=True, remove_stopwords=True)
    # 10 MB limit per shard forces multiple blocks to demonstrate the merge logic
    spimi = SPIMIIndexer(output_dir=index_dir, memory_limit_mb=10.0)
    parser = ReutersParser("")
    
    doc_count = 0
    for file_path in file_paths:
        for doc in parser.parse_file(file_path):
            doc_count += 1
            
            # Index Title
            title_tokens = preprocessor.process_with_positions(doc.title)
            for token, pos in title_tokens:
                spimi.add_token(token, doc.doc_id, pos)
            
            # Index Body (Offset positions by title length)
            title_len = len(doc.title.split()) if doc.title else 0
            body_tokens = preprocessor.process_with_positions(doc.body)
            for token, pos in body_tokens:
                spimi.add_token(token, doc.doc_id, pos + title_len)

            # Check memory constraint periodically
            if doc_count % 500 == 0 and spimi.is_memory_full():
                spimi.flush_to_disk()
                
    spimi.flush_to_disk()
    print(f"[Shard {shard_id}] Blocks flushed. Starting K-way merge and compression...")
    spimi.merge_blocks()
    print(f"[Shard {shard_id}] Index complete. Processed {doc_count} documents.")
    
    return shard_id, doc_count

def main():
    print("--- Starting Phase 2: Parallel SPIMI Index Construction ---")
    start_time = time.time()
    
    # Locate all 22 SGM files
    sgm_files = sorted(glob.glob(os.path.join("data", "raw", "reut2-*.sgm")))
    if not sgm_files:
        print("ERROR: No .sgm files found in data/raw/")
        return
        
    # Partition files into 4 shards
    num_shards = 4
    chunk_size = len(sgm_files) // num_shards + (1 if len(sgm_files) % num_shards > 0 else 0)
    
    shards = []
    for i in range(num_shards):
        chunk = sgm_files[i * chunk_size : (i + 1) * chunk_size]
        if chunk:
            shards.append((i, chunk))
            
    # Execute shards in parallel
    with Pool(processes=num_shards) as pool:
        results = pool.map(build_shard, shards)
        
    total_docs = sum(count for _, count in results)
    
    print("\n--- Indexing Complete ---")
    print(f"Successfully built 4 sharded indexes for {total_docs} documents.")
    print(f"Total multi-core build time: {time.time() - start_time:.2f} seconds.")
    print("Check the 'index_data' folder for the dictionary.json and postings.bin files in each shard.")

if __name__ == "__main__":
    main()