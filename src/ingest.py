import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Variáveis de configuração vindas do .env
PDF_PATH = os.getenv("PDF_PATH")  # caminho relativo/absoluto do PDF a ser processado
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL")  # modelo de embedding do Gemini
DATABASE_URL = os.getenv("DATABASE_URL")  # string de conexão com PostgreSQL
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")  # nome da coleção no PGVector

def ingest_pdf():
    # Define diretório base (pai do diretório atual do arquivo Python)
    current_dir = Path(__file__).parent.parent
    
    # Monta o caminho completo do PDF
    pdf_path = current_dir / PDF_PATH
    
    # Carrega o conteúdo do PDF como lista de documentos LangChain
    docs = PyPDFLoader(str(pdf_path)).load()

    # Divide o texto do PDF em chunks menores para facilitar embeddings
    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000,     # tamanho de cada pedaço
        chunk_overlap=150,   # sobreposição entre pedaços (contexto)
        add_start_index=False
    ).split_documents(docs)

    # Se não houver texto extraído, encerra
    if not splits:
        raise SystemExit(0)

    enriched = []
    # Ajusta metadados para remover chaves vazias
    for d in splits:
        meta = {k: v for k, v in d.metadata.items() if v not in ("", None)}
        new_doc = Document(
            page_content=d.page_content,  # conteúdo do chunk
            metadata=meta                 # metadados limpos
        )
        enriched.append(new_doc)
    
    # Cria IDs únicos para cada chunk/documento
    ids = [f"doc-{i}" for i in range(len(enriched))]

    # Instancia embeddings do Gemini
    embeddings = GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL)

    # Cria o vetor store no PostgreSQL (PGVector)
    store = PGVector(
        embeddings=embeddings,
        collection_name=PG_VECTOR_COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,  # armazena metadados em JSONB no Postgres
    )

    # Adiciona documentos + embeddings no banco
    store.add_documents(documents=enriched, ids=ids)

# Ponto de entrada do script
if __name__ == "__main__":
    ingest_pdf()