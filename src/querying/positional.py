from src.querying.index_reader import IndexReader

class PositionalSearchEngine:
    """Phrase and proximity queries over the positional postings."""
    def __init__(self, reader: IndexReader):
        self.reader = reader

    def search_phrase(self, terms: list[str]) -> list[int]:
        """Docs where the terms occur at consecutive positions (terms are already normalized)."""
        if not terms:
            return []
        postings = [self.reader.get_positions(t) for t in terms]
        common_docs = set(postings[0])
        for p in postings[1:]:
            common_docs &= set(p)

        results = []
        for doc in common_docs:
            # a start position s matches if term i occurs at s + i for every i
            starts = set(postings[0][doc])
            for i, p in enumerate(postings[1:], 1):
                starts &= {pos - i for pos in p[doc]}
                if not starts:
                    break
            if starts:
                results.append(doc)
        return sorted(results)

    def search_proximity(self, term1: str, term2: str, k: int) -> list[int]:
        """Finds documents where term1 and term2 occur within k positions of each other (either order)."""
        p1 = self.reader.get_positions(term1)
        p2 = self.reader.get_positions(term2)

        results = []
        common_docs = set(p1.keys()).intersection(set(p2.keys()))

        for doc in common_docs:
            pos1, pos2 = p1[doc], p2[doc]
            i = j = 0
            match_found = False

            # Linear merge of the two sorted position lists
            while i < len(pos1) and j < len(pos2):
                if abs(pos2[j] - pos1[i]) <= k:
                    match_found = True
                    break
                elif pos1[i] < pos2[j]:
                    i += 1
                else:
                    j += 1

            if match_found:
                results.append(doc)

        return sorted(results)
