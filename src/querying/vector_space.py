import math
import os
from collections import defaultdict
from src.parsing.preprocessor import TextPreprocessor
from src.querying.index_reader import IndexReader

class VectorSpaceModel:
    def __init__(self, shards_dir: str = "index_data", num_shards: int = 4):
        self.preprocessor = TextPreprocessor(use_stemming=True, remove_stopwords=True)
        self.shards = []
        universal_set = set()
        
        for i in range(num_shards):
            shard_path = os.path.join(shards_dir, f"shard_{i}")
            if os.path.exists(shard_path):
                reader = IndexReader(shard_path)
                self.shards.append(reader)
                universal_set.update(reader.get_all_doc_ids())
                
        self.N = len(universal_set) if universal_set else 1 # Total corpus size for IDF

    def get_term_stats(self, term: str) -> tuple[int, dict[int, int]]:
        """Aggregates Document Frequency (df) and Term Frequencies (tf) across all shards."""
        df = 0
        tf_dict = defaultdict(int)
        
        for shard in self.shards:
            if term in shard.dictionary:
                meta = shard.dictionary[term]
                df += meta["df"]
                for doc_id_str, tf in meta["tf"].items():
                    tf_dict[int(doc_id_str)] += tf
                    
        return df, dict(tf_dict)

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Scores documents against the query using tf-idf and cosine normalization."""
        query_tokens = self.preprocessor.process(query)
        if not query_tokens:
            return []
            
        # 1. Calculate Query Term Frequencies
        query_tf = defaultdict(int)
        for token in query_tokens:
            query_tf[token] += 1
            
        scores = defaultdict(float)
        doc_lengths = defaultdict(float)
        
        # 2. Compute lnc.ltc weights
        for token, q_tf in query_tf.items():
            df, doc_tfs = self.get_term_stats(token)
            if df == 0:
                continue
                
            # Query Weight (ltc): Logarithmic tf * inverse document frequency
            wt_q = (1 + math.log10(q_tf)) * math.log10(self.N / df)
            
            # Document Weight (lnc): Logarithmic tf * 1 (IDF is applied only to query to avoid double penalization)
            for doc_id, tf in doc_tfs.items():
                wt_d = 1 + math.log10(tf)
                scores[doc_id] += wt_q * wt_d
                doc_lengths[doc_id] += wt_d ** 2  # Accumulate squared weights for Cosine length
                
        # 3. Normalize Scores (Cosine Similarity)
        results = []
        for doc_id, dot_product in scores.items():
            normalized_score = dot_product / math.sqrt(doc_lengths[doc_id])
            results.append((doc_id, normalized_score))
            
        # 4. Sort and return Top K
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]