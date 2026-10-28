from app.config import USE_POSTGRES
from app.rag.store import list_documents

print(f"🗄️  Storage: {'Postgres (Neon)' if USE_POSTGRES else 'local files (Chroma + SQLite)'}" f" | {len(list_documents())} documents in the library\n")

from app.rag.ingest import ingest_file
from app.rag.store import get_vectorstore, list_documents


# Windows "Copy as path" adds quotes, so strip them
# path = input("📄 Path to a PDF, DOCX, TXT or MD file: ").strip().strip('"')
path = r"D:\\OneDrive\\Desktop\\Learn Agentic-AI with Projects\\Claude\\AI Research Assistant (Supervisor pattern)\\book.pdf"
print("\nIndexing...")
print(ingest_file(path))

print("\n📚 Library:")
for d in list_documents():
    print(f"  - {d['filename']}  ({d['pages']} pages, {d['chunks']} chunks)")


from app.rag.retrieval import format_source, retrieve
query = input("\n🔍 Ask something about the document: ")
for r in retrieve(query):
    print(f"\n--- {format_source(r)} | score {r['score']:.3f} ---")
    print(r["text"][:300], "...")