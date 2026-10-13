import math
import re
from src.querying.index_reader import IndexReader
from src.querying.positional import PositionalSearchEngine
from src.querying.wildcard import PermutermIndex


def intersect_with_skips(p1: list[int], p2: list[int], skip1: int = None, skip2: int = None):
    """Postings intersection with skip pointers (IIR 2.3, Fig. 2.10).
    skip = None -> sqrt(len) spacing (the textbook heuristic), skip = 0 -> no skip pointers.
    Returns (answer, number of docID comparisons)."""
    if skip1 is None:
        skip1 = int(math.sqrt(len(p1)))
    if skip2 is None:
        skip2 = int(math.sqrt(len(p2)))
    answer = []
    comparisons = 0
    i = j = 0
    while i < len(p1) and j < len(p2):
        comparisons += 1
        if p1[i] == p2[j]:
            answer.append(p1[i])
            i += 1
            j += 1
            continue
        comparisons += 1
        if p1[i] < p2[j]:
            # a skip pointer exists every skip1 positions
            if skip1 > 1 and i % skip1 == 0 and i + skip1 < len(p1):
                comparisons += 1
                if p1[i + skip1] <= p2[j]:
                    while i % skip1 == 0 and i + skip1 < len(p1) and p1[i + skip1] <= p2[j]:
                        i += skip1
                        comparisons += 1
                    continue
            i += 1
        else:
            if skip2 > 1 and j % skip2 == 0 and j + skip2 < len(p2):
                comparisons += 1
                if p2[j + skip2] <= p1[i]:
                    while j % skip2 == 0 and j + skip2 < len(p2) and p2[j + skip2] <= p1[i]:
                        j += skip2
                        comparisons += 1
                    continue
            j += 1
    return answer, comparisons


class BooleanSearchEngine:
    """Boolean retrieval: AND, OR, NOT, parentheses, "quoted phrases", proximity (a /k b) and wildcards (*).
    Adjacent operands without an operator are ANDed. By default it runs on the exact (unstemmed) index."""
    def __init__(self, index_dir: str = "index_data/exact"):
        self.reader = IndexReader(index_dir)
        self.preprocessor = self.reader.preprocessor
        self.positional = PositionalSearchEngine(self.reader)
        self._permuterm = None
        self.universal_set = self.reader.get_all_doc_ids()

    @property
    def permuterm(self):
        if self._permuterm is None:
            self._permuterm = PermutermIndex(self.reader.vocabulary())
        return self._permuterm

    def _get_term_postings(self, term: str) -> list[int]:
        """Postings of one query word (a wildcard expands to the union of all matching terms)."""
        if '*' in term:
            docs = set()
            for t in self.permuterm.search(term.lower()):
                docs.update(self.reader.get_postings(t))
            return sorted(docs)
        processed = self.preprocessor.process(term)
        if not processed:
            return []
        if len(processed) > 1:  # e.g. "u.s" tokenizes into two tokens -> treat as a phrase
            return self.positional.search_phrase(processed)
        return self.reader.get_postings(processed[0])

    def _single_token(self, term: str) -> str:
        processed = self.preprocessor.process(term)
        return processed[0] if processed else ""

    def tokenize_query(self, query: str) -> list:
        """Splits a query into operators and operands. Operands are ('term', w), ('phrase', w) or ('prox', w1, w2, k)."""
        raw = re.findall(r'\(|\)|"[^"]*"|/\d+|[^\s()"]+', query)
        tokens = []
        for tok in raw:
            if tok in ('AND', 'OR', 'NOT', '(', ')'):
                tokens.append(tok)
            elif tok.startswith('"'):
                tokens.append(('phrase', tok.strip('"')))
            elif re.fullmatch(r'/\d+', tok) and tokens and tokens[-1][0] == 'term':
                tokens.append(('prox_op', int(tok[1:])))
            elif tokens and tokens[-1][0] == 'prox_op':
                k = tokens.pop()[1]
                left = tokens.pop()[1]
                tokens.append(('prox', left, tok, k))
            else:
                tokens.append(('term', tok))

        # insert implicit ANDs between adjacent operands, e.g. "cocoa NOT drought" -> "cocoa AND NOT drought"
        result = []
        for tok in tokens:
            starts_operand = isinstance(tok, tuple) or tok in ('(', 'NOT')
            prev = result[-1] if result else None
            if starts_operand and prev is not None and (isinstance(prev, tuple) or prev == ')'):
                result.append('AND')
            result.append(tok)
        return result

    def _evaluate_operand(self, operand) -> list[int]:
        kind = operand[0]
        if kind == 'term':
            return self._get_term_postings(operand[1])
        if kind == 'phrase':
            return self.positional.search_phrase(self.preprocessor.process(operand[1]))
        if kind == 'prox':
            return self.positional.search_proximity(self._single_token(operand[1]), self._single_token(operand[2]), operand[3])
        return []

    def search(self, query: str) -> list[int]:
        """Evaluates a boolean query using Shunting-yard algorithm and RPN."""
        tokens = self.tokenize_query(query)

        precedence = {'NOT': 3, 'AND': 2, 'OR': 1, '(': 0, ')': 0}
        output = []
        operators = []

        # 1. Convert Infix to Postfix (Reverse Polish Notation)
        for token in tokens:
            if isinstance(token, tuple):
                output.append(token)
            elif token == 'NOT':
                operators.append(token)  # unary and right-associative: pops nothing
            elif token in ['AND', 'OR']:
                while operators and precedence[operators[-1]] >= precedence[token]:
                    output.append(operators.pop())
                operators.append(token)
            elif token == '(':
                operators.append(token)
            elif token == ')':
                while operators and operators[-1] != '(':
                    output.append(operators.pop())
                if operators:
                    operators.pop() # Discard the '('

        while operators:
            op = operators.pop()
            if op != '(':
                output.append(op)

        # 2. Evaluate Postfix Expression (operands are sorted doc ID lists)
        stack = []
        for token in output:
            if token == 'AND':
                if len(stack) < 2: continue
                p2, p1 = stack.pop(), stack.pop()
                stack.append(intersect_with_skips(p1, p2)[0])
            elif token == 'OR':
                if len(stack) < 2: continue
                p2, p1 = stack.pop(), stack.pop()
                stack.append(sorted(set(p1).union(p2)))
            elif token == 'NOT':
                if not stack: continue
                p1 = stack.pop()
                stack.append(sorted(self.universal_set - set(p1)))
            else:
                stack.append(self._evaluate_operand(token))

        return stack[0] if stack else []
