import os
from search import search_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

GOOGLE_MESSAGES = os.getenv("GOOGLE_MESSAGES")  # modelo de messages do Gemini

# mesmo template usado no search
PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
RESPOSTA: Formate com cabeçalho que fique legível, usando listas ou tabelas se necessário, com 4 linhas.
"""

def main():
    # pede entradas ao usuário
    print("=== Chat com Gemini e busca no PostgreSQL + PGVector ===")
    print("=== Entradas contexto e pergunta ===")
    # contexto = input("Digite o contexto: ").strip()
    # exemplo = Nomes de empresas que começam com a letra A, cujo faturamento seja maior que 1 milhão e que sejam as mais antigas.
    pergunta = input("Digite a pergunta: ").strip()

    # chama a função de busca (PGVector)
    results = search_prompt(pergunta)

    if not results:
        print("Não foi possível encontrar resultados.")
        return

    # exibe os documentos recuperados
    docs_text = []
    for i, (doc, score) in enumerate(results, start=1):
        snippet = doc.page_content[:300].replace("\n", " ")
        docs_text.append(snippet)

    # monta um contexto consolidado (docs recuperados)
    contexto_consolidado = "\n".join(docs_text)

    # cria o prompt para o chat
    chat_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # instancia o modelo Gemini
    chat = ChatGoogleGenerativeAI(model=GOOGLE_MESSAGES, temperature=0.1)

    # formata mensagens para o modelo
    messages = chat_prompt.format_messages(
        contexto=contexto_consolidado,
        pergunta=pergunta
    )

    # chama o modelo para gerar a resposta
    response = chat.invoke(messages)

    print("\n=== Resposta do Gemini ===")
    print(response.content)


if __name__ == "__main__":
    main()