import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL")
DATABASE_URL = os.getenv("DATABASE_URL")
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")

def ingest_pdf():
    current_dir = Path(__file__).parent.parent
    pdf_path = current_dir / PDF_PATH
    
    docs = PyPDFLoader(str(pdf_path)).load()

    splits = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=150, add_start_index=False).split_documents(docs)

    if not splits:
        raise SystemExit(0)

    enriched = []
    for d in splits:
        meta = {k: v for k, v in d.metadata.items() if v not in ("", None)}
        new_doc = Document(
            page_content=d.page_content,
            metadata=meta
        )
        enriched.append(new_doc)
    
    ids = [f"doc-{i}" for i in range(len(enriched))]

    embeddings = GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL)

    store = PGVector(
        embeddings=embeddings,
        collection_name=PG_VECTOR_COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )

    store.add_documents(documents=enriched, ids=ids)

if __name__ == "__main__":
    ingest_pdf()