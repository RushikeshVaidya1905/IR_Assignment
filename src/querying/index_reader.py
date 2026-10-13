import os
import json
from src.indexing.compress import VariableByteEncoder

class IndexReader:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        
        dict_path = os.path.join(index_dir, "dictionary.json")
        if os.path.exists(dict_path):
            with open(dict_path, 'r') as f:
                self.dictionary = json.load(f)
        else:
            self.dictionary = {}
            
        self.postings_path = os.path.join(index_dir, "postings.bin")

    def get_postings(self, term: str) -> set[int]:
        """Reads and decodes the postings list for a specific term."""
        if term not in self.dictionary:
            return set()
            
        meta = self.dictionary[term]
        
        with open(self.postings_path, 'rb') as f:
            f.seek(meta["offset"])
            encoded_bytes = f.read(meta["byte_size"])
            
        # Decode the binary bytes into gap values
        gaps = VariableByteEncoder.decode(encoded_bytes)
        # Reconstruct the original document IDs from the gaps
        doc_ids = VariableByteEncoder.gap_decode(gaps)
        
        return set(doc_ids)
        
    def get_all_doc_ids(self) -> set[int]:
        """Returns all known doc IDs in this shard (used for NOT queries)."""
        all_docs = set()
        for term_data in self.dictionary.values():
            all_docs.update(map(int, term_data["tf"].keys()))
        return all_docs