import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Variáveis configuradas no .env
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL")  # modelo de embedding do Gemini
DATABASE_URL = os.getenv("DATABASE_URL")  # string de conexão com PostgreSQL
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")  # nome da coleção dentro do PGVector

# Template para a prompt que será usado para montar a query
PROMPT_TEMPLATE = """
Recupere informações relevantes para responder à seguinte pergunta do usuário:

{pergunta}
"""

def search_prompt(pergunta: str):
    """
    Recebe uma pergunta,
    formata o prompt usando o template e
    executa busca semântica no PGVector.
    """

    # Monta o template substituindo {pergunta}
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    # Substitui as variáveis dentro do template
    query = prompt.format(
        pergunta=pergunta
    )

    # Cria embeddings usando Gemini
    embeddings = GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL)

    # Conecta no banco PostgreSQL com extensão PGVector
    store = PGVector(
        embeddings=embeddings,
        collection_name=PG_VECTOR_COLLECTION_NAME,  # nome da coleção para agrupar os vetores
        connection=DATABASE_URL,  # conexão vinda do .env
        use_jsonb=True,  # armazena metadados em JSONB no Postgres
    )

    # Busca semântica no vetor store
    # Retorna os documentos mais semelhantes ao "query" junto com a pontuação
    # k = define quantos resultados retornar
    results = store.similarity_search_with_score(query, k=10)

    return results
