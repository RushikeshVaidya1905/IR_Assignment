import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from src.parsing.reuters_parser import ReutersParser
from src.parsing.preprocessor import TextPreprocessor

def heaps_model(T, k, b):
    """Mathematical model for Heap's Law: M = k * T^b"""
    return k * (T ** b)

class LawFitter:
    def __init__(self, shards_dir="index_data", raw_dir="data/raw"):
        self.shards_dir = shards_dir
        self.raw_dir = raw_dir

    def plot_zipfs_law(self):
        """Fits and plots Zipf's Law using the term frequencies across all index shards."""
        term_freqs = {}
        
        # 1. Aggregate frequencies from all shards
        for i in range(4):
            dict_path = os.path.join(self.shards_dir, f"shard_{i}", "dictionary.json")
            if os.path.exists(dict_path):
                with open(dict_path, 'r') as f:
                    shard_dict = json.load(f)
                    for term, meta in shard_dict.items():
                        total_tf = sum(meta["tf"].values())
                        term_freqs[term] = term_freqs.get(term, 0) + total_tf
        
        if not term_freqs:
            print("Index is empty. Cannot plot Zipf's Law.")
            return

        # 2. Sort by frequency to calculate rank
        sorted_freqs = sorted(term_freqs.values(), reverse=True)
        ranks = np.arange(1, len(sorted_freqs) + 1)
        freqs = np.array(sorted_freqs)

        # 3. Fit Zipf's law in log-log space: log(f) = log(c) - s * log(r)
        log_ranks = np.log(ranks)
        log_freqs = np.log(freqs)
        
        slope, intercept = np.polyfit(log_ranks, log_freqs, 1)
        c = np.exp(intercept)
        s = -slope  # Zipf exponent

        # 4. Plotting
        plt.figure(figsize=(10, 5))
        plt.loglog(ranks, freqs, marker=".", linestyle="none", label="Actual Document Data")
        plt.loglog(ranks, c / (ranks ** s), 'r-', label=f"Zipf's Fit (s={s:.3f}, c={c:.2f})")
        
        plt.title("Zipf's Law on Reuters-21578")
        plt.xlabel("Rank (log scale)")
        plt.ylabel("Frequency (log scale)")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        
        plt.savefig("zipfs_law.png", dpi=300, bbox_inches='tight')
        print(f"-> Zipf's Law saved to 'zipfs_law.png' (Calculated Exponent s = {s:.3f})")

    def plot_heaps_law(self):
        """Streams documents to track vocabulary growth and fits Heap's Law."""
        parser = ReutersParser(self.raw_dir)
        preprocessor = TextPreprocessor(use_stemming=True, remove_stopwords=True)
        
        T_vals = [] # Total tokens
        M_vals = [] # Vocabulary size
        
        total_tokens = 0
        vocab = set()
        doc_count = 0
        
        # 1. Stream the corpus and record growth
        for doc in parser.parse_all():
            doc_count += 1
            tokens = preprocessor.process(doc.title + " " + doc.body)
            total_tokens += len(tokens)
            vocab.update(tokens)
            
            # Sample data points frequently (every 5 docs) for the graph
            if doc_count % 5 == 0:  
                T_vals.append(total_tokens)
                M_vals.append(len(vocab))
                
        T_vals = np.array(T_vals)
        M_vals = np.array(M_vals)

        if len(T_vals) < 2:
            print("Not enough data to fit Heap's Law.")
            return

        # 2. Fit non-linear curve M = k * T^b
        popt, _ = curve_fit(heaps_model, T_vals, M_vals, bounds=([0, 0], [np.inf, 1.0]))
        k, b = popt

        # 3. Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(T_vals, M_vals, marker="o", linestyle="none", label="Actual Vocabulary Growth")
        plt.plot(T_vals, heaps_model(T_vals, k, b), 'r-', label=f"Heap's Fit (k={k:.2f}, b={b:.3f})")
        
        plt.title("Heap's Law on Reuters-21578")
        plt.xlabel("Total Tokens Processed (T)")
        plt.ylabel("Vocabulary Size (M)")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        
        plt.savefig("heaps_law.png", dpi=300, bbox_inches='tight')
        print(f"-> Heap's Law saved to 'heaps_law.png' (Calculated k = {k:.2f}, b = {b:.3f})")