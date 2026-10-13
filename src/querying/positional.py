import os
from src.parsing.preprocessor import TextPreprocessor
from src.querying.index_reader import IndexReader

class PositionalSearchEngine:
    def __init__(self, shards_dir: str = "index_data", num_shards: int = 4):
        self.preprocessor = TextPreprocessor(use_stemming=True, remove_stopwords=True)
        self.shards = []
        for i in range(num_shards):
            shard_path = os.path.join(shards_dir, f"shard_{i}")
            if os.path.exists(shard_path):
                self.shards.append(IndexReader(shard_path))

    def _get_positions(self, term: str) -> dict[int, list[int]]:
        """Retrieves a mapping of DocID to a sorted list of term positions."""
        processed = self.preprocessor.process(term)
        if not processed:
            return {}
            
        token = processed[0]
        positions = {}
        for shard in self.shards:
            if token in shard.dictionary:
                meta = shard.dictionary[token]
                for doc_id_str, pos_list in meta["positions"].items():
                    doc_id = int(doc_id_str)
                    if doc_id not in positions:
                        positions[doc_id] = []
                    positions[doc_id].extend(pos_list)
                    positions[doc_id].sort()
        return positions

    def search_proximity(self, term1: str, term2: str, k: int) -> list[int]:
        """Finds documents where term2 occurs within k positions after term1."""
        p1 = self._get_positions(term1)
        p2 = self._get_positions(term2)
        
        results = []
        common_docs = set(p1.keys()).intersection(set(p2.keys()))
        
        for doc in common_docs:
            pos1, pos2 = p1[doc], p2[doc]
            i = j = 0
            match_found = False
            
            # Linear intersection algorithm for positional lists
            while i < len(pos1) and j < len(pos2):
                diff = pos2[j] - pos1[i]
                if 0 < diff <= k:
                    match_found = True
                    break
                elif pos1[i] < pos2[j]:
                    i += 1
                else:
                    j += 1
                    
            if match_found:
                results.append(doc)
                
        return sorted(results)