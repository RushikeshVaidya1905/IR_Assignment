import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK stopwords quietly on first run
nltk.download('stopwords', quiet=True)

class TextPreprocessor:
    def __init__(self, use_stemming: bool = True, remove_stopwords: bool = True):
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def tokenize(self, text: str) -> list[str]:
        """Splits text into alphanumeric tokens, removing pure punctuation."""
        # Finds contiguous alphanumeric strings
        return re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())

    def process(self, text: str) -> list[str]:
        """Applies full normalization pipeline to a string."""
        tokens = self.tokenize(text)
        processed_tokens = []
        
        for token in tokens:
            if self.remove_stopwords and token in self.stop_words:
                continue
            
            if self.use_stemming:
                token = self.stemmer.stem(token)
            
            processed_tokens.append(token)

        return processed_tokens

    def process_with_positions(self, text: str) -> list[tuple[str, int]]:
        """Returns tuples of (token, position_index) for positional index construction."""
        raw_tokens = self.tokenize(text)
        results = []
        
        for pos, token in enumerate(raw_tokens):
            if self.remove_stopwords and token in self.stop_words:
                continue
            
            if self.use_stemming:
                token = self.stemmer.stem(token)
                
            results.append((token, pos))
            
        return results