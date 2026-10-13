import os
import json
import psutil
from collections import defaultdict
from src.indexing.compress import VariableByteEncoder

class SPIMIIndexer:
    def __init__(self, output_dir: str, memory_limit_mb: float = 20.0):
        self.output_dir = output_dir
        self.memory_limit_mb = memory_limit_mb
        self.dictionary = defaultdict(lambda: defaultdict(list))
        self.block_number = 0
        os.makedirs(output_dir, exist_ok=True)
        
    def is_memory_full(self) -> bool:
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        return mem_mb > self.memory_limit_mb
        
    def add_token(self, term: str, doc_id: int, position: int):
        self.dictionary[term][str(doc_id)].append(position)
        
    def flush_to_disk(self):
        if not self.dictionary:
            return
            
        block_file = os.path.join(self.output_dir, f"block_{self.block_number}.json")
        sorted_terms = sorted(self.dictionary.keys())
        block_data = {term: self.dictionary[term] for term in sorted_terms}
            
        with open(block_file, 'w') as f:
            json.dump(block_data, f)
            
        print(f"-> Flushed {os.path.basename(block_file)} ({len(sorted_terms)} terms).")
        self.block_number += 1
        self.dictionary.clear() 

    def get_current_memory_usage(self) -> float:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    def merge_blocks(self, final_dict_name="dictionary.json", final_postings_name="postings.bin"):
        """Merges partial blocks, applies VB compression, and writes the final index."""
        block_files = [os.path.join(self.output_dir, f"block_{i}.json") for i in range(self.block_number)]
        merged_dict = defaultdict(lambda: defaultdict(list))
        
        # Load and combine all partial blocks
        for file_path in block_files:
            if not os.path.exists(file_path):
                continue
            with open(file_path, 'r') as f:
                block_data = json.load(f)
                for term, docs in block_data.items():
                    for doc_id, positions in docs.items():
                        merged_dict[term][doc_id].extend(positions)
                        merged_dict[term][doc_id].sort()

        final_dictionary = {}
        postings_file_path = os.path.join(self.output_dir, final_postings_name)
        
        with open(postings_file_path, 'wb') as f_out:
            for term in sorted(merged_dict.keys()):
                # Sort doc IDs numerically
                doc_ids = sorted([int(d) for d in merged_dict[term].keys()])
                
                # Gap encode and compress
                gapped_docs = VariableByteEncoder.gap_encode(doc_ids)
                encoded_bytes = bytearray()
                for gap in gapped_docs:
                    encoded_bytes.extend(VariableByteEncoder.encode_number(gap))
                
                # Write to binary file and record offset/size
                offset = f_out.tell()
                f_out.write(encoded_bytes)
                byte_size = f_out.tell() - offset
                
                # Store metadata, term frequencies (for vector space scoring), and positions (for phrase queries)
                final_dictionary[term] = {
                    "df": len(doc_ids),
                    "offset": offset,
                    "byte_size": byte_size,
                    "tf": {d: len(merged_dict[term][str(d)]) for d in doc_ids},
                    "positions": {d: merged_dict[term][str(d)] for d in doc_ids}
                }

        dict_path = os.path.join(self.output_dir, final_dict_name)
        with open(dict_path, 'w') as f:
            json.dump(final_dictionary, f)
            
        # Clean up temporary JSON blocks
        for file_path in block_files:
            if os.path.exists(file_path):
                os.remove(file_path)