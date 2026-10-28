import hashlib
from app.rag.store import (add_chunks, add_document_record, delete_chunks, get_document, remove_document_record)
from pathlib import Path
import re

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_CHARS

from langchain_community.document_loaders import Docx2txtLoader, PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.rag.store import (add_document_record, get_document, get_vectorstore, remove_document_record, set_shared) #type: ignore

SUPPORTED_TYPES = {".pdf", ".docx", ".txt", ".md"}

def _clean_text(text: str) -> str:
    """Fix common PDF extraction artifacts so keyword search sees whole words."""
    text = text.replace("\x00", "")  # Postgres can't store NUL characters    
    text = re.sub(r"(\w)-\n(\w)", r"\1\2",
                  text)     # re-join words hyphenated across lines
    # single line breaks -> spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # collapse repeated spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_VENUE = re.compile(r"arXiv|Proceedings|Journal|Conference|Advances in|pp\.|Vol\.|"
                    r"ICML|NIPS|NeurIPS|ICLR|CVPR", re.IGNORECASE)
_NUMBER = re.compile(r"\b\d{1,4}\b")


def _is_low_value(text: str) -> bool:
    """Bibliography entries and index pages match keywords well but explain nothing."""
    # Reference lists: many years plus venue names (journals, conferences, arXiv)
    if len(_YEAR.findall(text)) >= 6 and len(_VENUE.findall(text)) >= 3:
        return True
    # Back-of-book index: mostly page numbers ("dropout, 252, 258, 265 ...")
    words = text.split()
    return len(words) > 20 and len(_NUMBER.findall(text)) / len(words) > 0.4

def _file_id(path: Path, owner_id: str | None = None) -> str:
    """Fingerprint of the file's content plus its owner, so each user gets their own copy."""
    digest = hashlib.sha256()
    if owner_id:
        digest.update(owner_id.encode("utf-8"))
    digest.update(path.read_bytes())
    return digest.hexdigest()[:16]

def load_pages(path: Path, ext: str) -> list[Document]:
    """Load a file as a list of pages. Only PDFs have real page numbers."""
    if ext == ".pdf":
        pages = PyMuPDFLoader(str(path)).load()  # one Document per page
        # PyMuPDF counts pages from 0; people count from 1
        return [Document(page_content=_clean_text(p.page_content),
                         metadata={"page": p.metadata.get("page", 0) + 1})
                for p in pages]

    if ext == ".docx":
        docs = Docx2txtLoader(str(path)).load()
    elif ext in (".txt", ".md"):
        docs = TextLoader(str(path), encoding="utf-8").load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Word and text files have no page numbers; page 0 means "not paginated"
    return [Document(page_content=_clean_text(d.page_content), metadata={"page": 0}) for d in docs]


def split_into_chunks(pages: list[Document], doc_id: str, filename: str) -> list[Document]:
    """Split page by page, so every chunk knows exactly which page it came from."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for page in pages:
        for piece in splitter.split_text(page.page_content):
            if len(piece.strip()) >= MIN_CHUNK_CHARS and not _is_low_value(piece):
                chunks.append(Document(
                    page_content=piece,
                    metadata={
                        "doc_id": doc_id,
                        "filename": filename,
                        "page": page.metadata["page"],
                        "chunk_index": len(chunks),
                    },
                ))
    return chunks


def _looks_garbled(chunks: list[Document]) -> bool:
    """If spaces are missing, 'words' become very long. Normal text averages ~5 characters."""
    words = " ".join(c.page_content for c in chunks[:200]).split()
    return bool(words) and sum(len(w) for w in words) / len(words) > 10


def ingest_file(file_path: str | Path, filename: str | None = None, owner_id: str | None = None) -> dict:
    """Load, chunk, embed and store one file. Returns a summary."""
    path = Path(file_path)
    filename = filename or path.name
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported file type '{ext}'. "
                         f"Supported: {', '.join(sorted(SUPPORTED_TYPES))}")

    doc_id = _file_id(path, owner_id)
    existing = get_document(doc_id)
    if existing:
        return {**existing, "status": "already in library"}

    pages = load_pages(path, ext)
    chunks = split_into_chunks(pages, doc_id, filename)
    if not chunks:
        raise ValueError("No text could be extracted. If this is a scanned PDF "
                         "(images of pages), it would need OCR, which isn't supported yet.")

    if _looks_garbled(chunks):
        raise ValueError("The extracted text looks garbled (words joined without spaces), "
                         "so search results would be poor. Try a different copy of this file.")


    # Stable IDs like "<doc_id>-0", "<doc_id>-1" ... make deleting easy later
    ids = [f"{doc_id}-{c.metadata['chunk_index']}" for c in chunks]
    add_chunks(chunks, ids)  # embeds and stores: Chroma locally, Postgres online

    page_count = len(pages) if ext == ".pdf" else 0
    add_document_record(doc_id, filename, ext.lstrip("."), page_count, len(chunks), owner_id)
    return {"doc_id": doc_id, "filename": filename, "pages": page_count,
            "chunks": len(chunks), "status": "indexed"}

def delete_document(doc_id: str, user_id: str | None = None) -> None:
    """Remove a document's chunks and registry entry. Users may delete only their own."""
    record = get_document(doc_id)
    if not record:
        return
    if user_id is not None and record["owner_id"] != user_id:
        raise PermissionError("You can only delete your own documents.")
    ids = [f"{doc_id}-{i}" for i in range(record["chunks"])]
    delete_chunks(doc_id, ids)
    remove_document_record(doc_id)


def share_document(doc_id: str, shared: bool, user: dict) -> None:
    """Only the admin may share, and only their own documents."""
    record = get_document(doc_id)
    if not record or not user.get("is_admin") or record["owner_id"] != user["user_id"]:
        raise PermissionError("Only the admin can share their own documents.")
    set_shared(doc_id, shared)