from src.querying.index_reader import IndexReader

class SpellChecker:
    """Isolated-term correction (IIR 3.3): vocabulary terms within edit distance 2, closest first,
    ties broken by document frequency (more common terms are more likely intended).
    Uses the exact (unstemmed) index so suggestions are real words."""
    def __init__(self, reader: IndexReader = None):
        self.reader = reader or IndexReader("index_data/exact")
        self.vocabulary = self.reader.vocabulary()

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Minimum number of single-character edits (insert, delete, substitute, or swap two adjacent
        characters - Damerau's extension) required to change s1 into s2."""
        d = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
        for i in range(len(s1) + 1):
            d[i][0] = i
        for j in range(len(s2) + 1):
            d[0][j] = j
        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                d[i][j] = min(d[i - 1][j] + 1,         # deletion
                              d[i][j - 1] + 1,         # insertion
                              d[i - 1][j - 1] + cost)  # substitution
                if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                    d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)  # transposition
        return d[len(s1)][len(s2)]

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
                
        # Sort by distance (ascending), then by document frequency (descending), then alphabetically
        suggestions.sort(key=lambda x: (x[1], -self.reader.df(x[0]), x[0]))
        return [term for term, dist in suggestions[:5]] # Return top 5 suggestions