import os
import time
import shutil
import tracemalloc
from multiprocessing import Pool
from src.parsing.corpus_parser import CorpusParser
from src.parsing.preprocessor import TextPreprocessor
from src.indexing.spimi import SPIMIIndexer, BSBIIndexer, merge_blocks, analyze_document
from src.indexing.index_format import IndexWriter

# The two indexes Lexis uses:
#   exact  : no stemming, no stop words -> Boolean / phrase / proximity / wildcard / spelling (matches a brute-force scan)
#   ranked : stemming + stop words      -> free-text ranked retrieval
INDEXES = {
    "index_data/exact": dict(use_stemming=False, remove_stopwords=False),
    "index_data/ranked": dict(use_stemming=True, remove_stopwords=True),
}

def build_shard(shard_info):
    """Worker function: inverts a subset of the corpus into sorted blocks on disk (SPIMI or BSBI)."""
    shard_id, file_paths, tmp_dir, config, memory_limit, method, measure_memory = shard_info
    if measure_memory:
        tracemalloc.start()
    start = time.time()

    preprocessor = TextPreprocessor(**config)
    indexer_class = BSBIIndexer if method == "bsbi" else SPIMIIndexer
    indexer = indexer_class(output_dir=tmp_dir, memory_limit=memory_limit, block_prefix=f"shard{shard_id}_block")
    parser = CorpusParser()

    docs = {}
    for doc in parser.parse_files(file_paths):
        tokens, docs[doc.doc_id] = analyze_document(preprocessor, doc)
        for token, pos in tokens:
            indexer.add_token(token, doc.doc_id, pos)

        # Dummy memory constraint: flush a sorted block when the in-memory postings limit is reached
        if indexer.is_memory_full():
            indexer.flush_to_disk()

    indexer.flush_to_disk()
    peak_memory = tracemalloc.get_traced_memory()[1] if measure_memory else 0
    if measure_memory:
        tracemalloc.stop()
    return dict(shard_id=shard_id, block_files=indexer.block_files, docs=docs,
                bytes_written=indexer.bytes_written, peak_memory=peak_memory, time=time.time() - start)

def build_index(out_dir, use_stemming, remove_stopwords, num_shards=4, memory_limit=100_000,
                method="spimi", measure_memory=False, verbose=True):
    """Streams the corpus in `num_shards` parallel workers, then k-way merges all blocks into one index.
    Returns construction statistics (used in the report)."""
    start_time = time.time()
    config = dict(use_stemming=use_stemming, remove_stopwords=remove_stopwords)
    files = CorpusParser().list_files()
    if not files:
        raise FileNotFoundError("No doc_*.txt files found in data/corpus/")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    tmp_dir = os.path.join(out_dir, "blocks")

    # Partition the files into contiguous shards
    chunk_size = -(-len(files) // num_shards)
    shards = [(i, files[i * chunk_size:(i + 1) * chunk_size], tmp_dir, config, memory_limit, method, measure_memory)
              for i in range(num_shards)]

    # Phase 1: parallel inversion into sorted blocks
    if num_shards == 1:
        results = [build_shard(shards[0])]
    else:
        with Pool(processes=num_shards) as pool:
            results = pool.map(build_shard, shards)
    inversion_time = time.time() - start_time

    # Phase 2: merge every block of every shard into one index
    docs = {}
    block_files = []
    for r in results:
        docs.update(r["docs"])
        block_files.extend(r["block_files"])
    writer = IndexWriter(out_dir)
    merge_blocks(block_files, writer)
    index_bytes = writer.close(docs, config)
    os.rmdir(tmp_dir)

    stats = dict(docs=len(docs), blocks=len(block_files), terms=len(writer.entries),
                 bytes_written=sum(r["bytes_written"] for r in results), index_bytes=index_bytes,
                 peak_memory=max(r["peak_memory"] for r in results),
                 inversion_time=inversion_time, worker_time=max(r["time"] for r in results),
                 total_time=time.time() - start_time)
    if verbose:
        print(f"[{out_dir}] {stats['docs']} docs, {stats['terms']} terms, {stats['blocks']} blocks "
              f"({stats['bytes_written'] / 1e6:.1f} MB written), index {index_bytes / 1e6:.1f} MB, "
              f"{stats['total_time']:.2f} s")
    return stats

def main():
    print("--- Building Lexis indexes (4 parallel shards, SPIMI, k-way merge) ---")
    for out_dir, config in INDEXES.items():
        build_index(out_dir, **config)

if __name__ == "__main__":
    main()
