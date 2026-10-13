import re
from bisect import bisect_left

class PermutermIndex:
    """Permuterm index (IIR 3.2.1) over the vocabulary of an index."""
    def __init__(self, vocabulary):
        self.vocabulary = set(vocabulary)

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
        """Rotates a wildcard query so that a '*' is at the end.
        X*Y -> Y$X ; for several stars X*Y*Z only the outer parts are used (Z$X), then results are filtered."""
        parts = query.split('*')
        # Example: 'he*o' -> 'he', 'o' -> 'o$he'
        return parts[-1] + "$" + parts[0]

    def search(self, wildcard_query: str) -> list[str]:
        """Finds all vocabulary terms matching the wildcard query (one or more '*')."""
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

        # Post-filtering for the middle parts of multi-star queries (e.g. pet*chem*)
        if wildcard_query.count('*') > 1:
            pattern = re.compile('^' + '.*'.join(map(re.escape, wildcard_query.split('*'))) + '$')
            matches = {m for m in matches if pattern.match(m)}

        return sorted(list(matches))
