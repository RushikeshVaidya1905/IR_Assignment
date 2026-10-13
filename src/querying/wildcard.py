import os
import json
from bisect import bisect_left

class PermutermIndex:
    def __init__(self, shards_dir: str = "index_data", num_shards: int = 4):
        self.vocabulary = set()
        
        # Load unique vocabulary across all shards
        for i in range(num_shards):
            dict_path = os.path.join(shards_dir, f"shard_{i}", "dictionary.json")
            if os.path.exists(dict_path):
                with open(dict_path, 'r') as f:
                    shard_dict = json.load(f)
                    self.vocabulary.update(shard_dict.keys())
                    
        # Build the Permuterm Index: rotated_string -> original_term
        self.index = {}
        for term in self.vocabulary:
            term_dollar = term + "$"
            for i in range(len(term_dollar)):
                rotated = term_dollar[i:] + term_dollar[:i]
                self.index[rotated] = term
                
        # Sorted keys for fast prefix binary searching
        self.sorted_keys = sorted(self.index.keys())

    def get_permuterm_prefix(self, query: str) -> str:
        """Rotates a wildcard query so the '*' is at the end."""
        parts = query.split('*')
        if len(parts) != 2:
            return query # Not a single-wildcard query
            
        # Example: 'he*o' -> 'he', 'o' -> 'o$he*'
        return parts[1] + "$" + parts[0]

    def search(self, wildcard_query: str) -> list[str]:
        """Finds all vocabulary terms matching the wildcard query."""
        if '*' not in wildcard_query:
            return [wildcard_query] if wildcard_query in self.vocabulary else []

        prefix = self.get_permuterm_prefix(wildcard_query)
        matches = set()
        
        # Binary search to find the first matching prefix
        idx = bisect_left(self.sorted_keys, prefix)
        
        # Scan forward from the insertion point
        while idx < len(self.sorted_keys):
            key = self.sorted_keys[idx]
            if not key.startswith(prefix):
                break
            matches.add(self.index[key])
            idx += 1
            
        return sorted(list(matches))