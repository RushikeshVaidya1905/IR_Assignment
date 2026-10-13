import os
import json

class SpellChecker:
    def __init__(self, shards_dir: str = "index_data", num_shards: int = 4):
        self.vocabulary = []
        
        for i in range(num_shards):
            dict_path = os.path.join(shards_dir, f"shard_{i}", "dictionary.json")
            if os.path.exists(dict_path):
                with open(dict_path, 'r') as f:
                    shard_dict = json.load(f)
                    self.vocabulary.extend(shard_dict.keys())
                    
        self.vocabulary = list(set(self.vocabulary))

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Computes the minimum number of single-character edits required to change s1 into s2."""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    def suggest(self, query_term: str, max_distance: int = 2) -> list[str]:
        """Returns vocabulary terms within a maximum edit distance."""
        if query_term in self.vocabulary:
            return [query_term] # Word is already correct
            
        suggestions = []
        # Optimization: Only check terms with a similar length
        candidates = [v for v in self.vocabulary if abs(len(v) - len(query_term)) <= max_distance]
        
        for vocab_term in candidates:
            dist = self.levenshtein_distance(query_term, vocab_term)
            if dist <= max_distance:
                suggestions.append((vocab_term, dist))
                
        # Sort by distance (ascending) then alphabetically
        suggestions.sort(key=lambda x: (x[1], x[0]))
        return [term for term, dist in suggestions[:5]] # Return top 5 suggestions