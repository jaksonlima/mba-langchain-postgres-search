import os
from search import search_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate as CoreChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import trim_messages

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

GOOGLE_MESSAGES = os.getenv("GOOGLE_MESSAGES")  # modelo de messages do Gemini

# prompt template context
PROMPT_TEMPLATE_CONTEXT = """
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
"""

# prompt template usuario
PROMPT_TEMPLATE_USER = """
PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
RESPOSTA: Formate com cabeçalho que fique legível, usando listas ou tabelas se necessário, com 4 linhas.
"""

session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]


def prepare_inputs(payload: dict) -> dict:
    FALLBACK = "Não tenho informações necessárias para responder sua pergunta"
    raw_history = payload.get("raw_history", [])

    # 1) Remove mensagens de fallback do assistente
    filtered = []
    for m in raw_history:
        role = getattr(m, "type", None) or getattr(m, "role", None) or ""
        content = getattr(m, "content", "")
        if role in ("ai", "assistant") and FALLBACK in content:
            continue
        filtered.append(m)

    # 2) Apara o histórico (mantém só o mais recente)
    trimmed = trim_messages(
        filtered,
        token_counter=len,     # aqui só conta mensagens; se quiser tokens reais, pode plugar um contador de tokens
        max_tokens=4,          # ajusta a quantidade que deseja manter
        strategy="last",
        start_on="human",
        include_system=True,
        allow_partial=False,
    )

    # 3) Retorna no formato que seu prompt espera
    return {
        "pergunta": payload.get("pergunta", ""),
        "contexto": payload.get("contexto", ""),
        "history": trimmed,
    }

def main():
    # cria o prompt para o chat (com histórico) — contexto/regras e pergunta como 'human'
    chat_prompt = CoreChatPromptTemplate.from_messages([
        ("system", PROMPT_TEMPLATE_CONTEXT),
        MessagesPlaceholder(variable_name="history"),
        ("human", PROMPT_TEMPLATE_USER),
    ])

    # instancia o modelo Gemini
    chat = ChatGoogleGenerativeAI(model=GOOGLE_MESSAGES, temperature=0.1)

    prepare = RunnableLambda(prepare_inputs)

    # encadeia prompt -> modelo
    # chain = chat_prompt | chat
    chain = prepare | chat_prompt | chat

    # envolve com histórico por sessão
    conversational_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="pergunta",   # campo que representa a entrada atual do usuário
        history_messages_key="raw_history",  # placeholder do histórico
    )

    # pede entradas ao usuário
    print("\n=== Chat com Gemini e busca no PostgreSQL + PGVector ===")
    print("=== Entradas contexto e pergunta ===")
    while True:
        # exemplo = Nomes de empresas que começam com a letra A, cujo faturamento seja maior que 1 milhão e que sejam as mais antigas.
        pergunta = input("\nDigite a pergunta (ou 'sair'): ").strip()

        print("\n---------------------------")
        print("Historico da sessão (demo-session):")
        print(get_session_history("demo-session"))  # exibe o histórico de mensagens da sessão
        print("\n---------------------------")

        if pergunta.lower() in {"sair", "exit", "quit"}:
            print("\nEncerrando. Até mais!\n")
            break
        if not pergunta:
            continue

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

         # chama o modelo para gerar a resposta (com histórico da sessão)
        config = {"configurable": {"session_id": "demo-session"}} # id da sessão (pode ser dinâmico)
        response = conversational_chain.invoke(
            {"contexto": contexto_consolidado, "pergunta": pergunta},
            config=config
        )

        print("\n=== Resposta do Gemini ===\n")
        print("📌\n")
        print(response.content)
        print("\n---------------------------")

if __name__ == "__main__":
    main()