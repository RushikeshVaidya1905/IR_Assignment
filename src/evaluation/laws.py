import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from src.parsing.corpus_parser import CorpusParser
from src.querying.index_reader import IndexReader

# Textbook values for Reuters-RCV1 (IIR 5.1): Heaps k = 44, b = 0.49; Zipf cf_i proportional to 1/i (exponent 1)
RCV1_HEAPS_K, RCV1_HEAPS_B, RCV1_ZIPF_S = 44, 0.49, 1.0

def heaps_model(T, k, b):
    """Mathematical model for Heap's Law: M = k * T^b"""
    return k * (T ** b)

class LawFitter:
    """Vocabulary laws on the exact index (case folding only, no stemming or stop word removal)."""
    def __init__(self, index_dir="index_data/exact", out_dir="results"):
        self.reader = IndexReader(index_dir)
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def plot_zipfs_law(self):
        """Fits and plots Zipf's Law using the collection frequency of every term in the index."""
        # 1. Collection frequency (cf) of every term
        term_freqs = {}
        for term in self.reader.vocabulary():
            term_freqs[term] = sum(len(p) for p in self.reader.get_positions(term).values())
        self.reader._cache.clear()
        
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
        
        plt.title("Zipf's Law on the Lexis corpus (Reuters-21578)")
        plt.xlabel("Rank (log scale)")
        plt.ylabel("Frequency (log scale)")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        
        plt.savefig(os.path.join(self.out_dir, "zipfs_law.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"-> Zipf's Law saved to '{self.out_dir}/zipfs_law.png' (Calculated Exponent s = {s:.3f}, RCV1: {RCV1_ZIPF_S})")
        return {"zipf_s": s, "zipf_c": c, "rcv1_s": RCV1_ZIPF_S}

    def plot_heaps_law(self):
        """Streams documents to track vocabulary growth and fits Heap's Law."""
        parser = CorpusParser()
        preprocessor = self.reader.preprocessor
        
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
        
        plt.title("Heap's Law on the Lexis corpus (Reuters-21578)")
        plt.xlabel("Total Tokens Processed (T)")
        plt.ylabel("Vocabulary Size (M)")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        
        plt.savefig(os.path.join(self.out_dir, "heaps_law.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"-> Heap's Law saved to '{self.out_dir}/heaps_law.png' (Calculated k = {k:.2f}, b = {b:.3f}; "
              f"RCV1: k = {RCV1_HEAPS_K}, b = {RCV1_HEAPS_B})")
        return {"heaps_k": k, "heaps_b": b, "rcv1_k": RCV1_HEAPS_K, "rcv1_b": RCV1_HEAPS_B,
                "tokens": int(total_tokens), "vocabulary": len(vocab)}