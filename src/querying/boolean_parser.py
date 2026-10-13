import os
import re
from src.parsing.preprocessor import TextPreprocessor
from src.querying.index_reader import IndexReader

class BooleanSearchEngine:
    def __init__(self, shards_dir: str = "index_data", num_shards: int = 4):
        self.preprocessor = TextPreprocessor(use_stemming=True, remove_stopwords=True)
        self.shards = []
        self.universal_set = set()
        
        # Load all shards
        for i in range(num_shards):
            shard_path = os.path.join(shards_dir, f"shard_{i}")
            if os.path.exists(shard_path):
                reader = IndexReader(shard_path)
                self.shards.append(reader)
                self.universal_set.update(reader.get_all_doc_ids())

    def _get_term_postings(self, term: str) -> set[int]:
        """Gets processed term postings by querying all shards."""
        processed = self.preprocessor.process(term)
        if not processed:
            return set()
            
        query_token = processed[0]
        docs = set()
        for shard in self.shards:
            docs.update(shard.get_postings(query_token))
        return docs

    def search(self, query: str) -> list[int]:
        """Evaluates a boolean query using Shunting-yard algorithm and RPN."""
        # Tokenize preserving parenthesis and uppercase boolean operators
        tokens = re.findall(r'\(|\)|AND|OR|NOT|\w+', query)
        
        precedence = {'NOT': 3, 'AND': 2, 'OR': 1, '(': 0, ')': 0}
        output = []
        operators = []
        
        # 1. Convert Infix to Postfix (Reverse Polish Notation)
        for token in tokens:
            if token in ['AND', 'OR', 'NOT']:
                while operators and precedence[operators[-1]] >= precedence[token]:
                    output.append(operators.pop())
                operators.append(token)
            elif token == '(':
                operators.append(token)
            elif token == ')':
                while operators and operators[-1] != '(':
                    output.append(operators.pop())
                operators.pop() # Discard the '('
            else:
                output.append(token)
                
        while operators:
            output.append(operators.pop())
            
        # 2. Evaluate Postfix Expression
        stack = []
        for token in output:
            if token == 'AND':
                if len(stack) < 2: continue
                set2, set1 = stack.pop(), stack.pop()
                stack.append(set1.intersection(set2))
            elif token == 'OR':
                if len(stack) < 2: continue
                set2, set1 = stack.pop(), stack.pop()
                stack.append(set1.union(set2))
            elif token == 'NOT':
                if not stack: continue
                set1 = stack.pop()
                stack.append(self.universal_set - set1)
            else:
                stack.append(self._get_term_postings(token))
                
        return sorted(list(stack[0])) if stack else []