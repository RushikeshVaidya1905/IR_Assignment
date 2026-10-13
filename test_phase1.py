from src.parsing.reuters_parser import ReutersParser
from src.parsing.preprocessor import TextPreprocessor

def main():
    raw_dir = "data/raw"
    parser = ReutersParser(raw_dir)
    preprocessor = TextPreprocessor(use_stemming=True, remove_stopwords=True)

    print("--- Testing Phase 1 Pipeline ---")
    
    doc_count = 0
    # We will just break after the first document to verify output
    for doc in parser.parse_all():
        doc_count += 1
        
        print(f"Document ID: {doc.doc_id}")
        print(f"Raw Title: {doc.title}")
        print(f"Processed Title Tokens: {preprocessor.process(doc.title)}\n")
        
        print("Processed Body Positions (First 15 tokens):")
        print(preprocessor.process_with_positions(doc.body)[:15])
        break # Remove this break if you want to test parsing all 10,788 docs

    if doc_count == 0:
        print("ERROR: No documents found. Ensure .sgm files are in data/raw/")
    else:
        print("\nPhase 1 Pipeline successfully verified.")

if __name__ == "__main__":
    main()