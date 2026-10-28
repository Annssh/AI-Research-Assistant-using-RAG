from app.config import USE_POSTGRES
from app.rag.store import list_documents

print(f"🗄️  Storage: {'Postgres (Neon)' if USE_POSTGRES else 'local files (Chroma + SQLite)'}" f" | {len(list_documents())} documents in the library\n")

import time

from app.rag.retrieval import format_source, retrieve, rewrite_query
from app.config import MIN_RELEVANCE

question = input("🔍 Question: ").strip()

queries = rewrite_query(question)
print("\n✏️  Rewritten queries:")
for q in queries:
    print("   -", q)

print("\n=== A) Plain vector search ===")
from app.rag.retrieval import _get_index, vector_search
by_id = _get_index()["by_id"]
for chunk_id in vector_search(question, 5):
    text, m = by_id[chunk_id]
    print(f"   p.{m['page']:<4} {text[:100]!r}")


print("\n=== B) Rewrite + hybrid + rerank (Phase R2) ===")
start = time.time()
# min_relevance=0 shows every candidate, so we can see the scores of weak matches too
results = retrieve(question, queries=queries, min_relevance=0.0)
for r in results:
    kept = "✅" if r["score"] >= MIN_RELEVANCE else "🚫"
    print(
        f"   {kept} {format_source(r):<18} score {r['score']:.3f}  {r['text'][:70]!r}")
if not any(r["score"] >= MIN_RELEVANCE for r in results):
    print("   → No relevant passages: the documents probably don't cover this question.")
print(f"\n⏱  Retrieval took {time.time() - start:.1f}s")
