from src.evaluation.laws import LawFitter

def main():
    print("--- Starting Phase 4: Data Analytics ---")
    fitter = LawFitter()
    
    print("\nCalculating Zipf's Law from index...")
    fitter.plot_zipfs_law()
    
    print("\nCalculating Heap's Law (this requires streaming the raw text again)...")
    fitter.plot_heaps_law()

    print("\nDone! Check your project folder for zipfs_law.png and heaps_law.png to include in your final report.")

if __name__ == "__main__":
    main()