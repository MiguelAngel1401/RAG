import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain import hub
from langgraph.graph import START, END, StateGraph
from typing_extensions import List, TypedDict, NotRequired
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState

# Cargar entorno
load_dotenv()
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["USER_AGENT"] = os.getenv("USER_AGENT")

st.set_page_config(page_title="Chat RAG PDF", layout="wide")
st.title("🧠 Chat RAG con Streamlit y LangChain")

# Subida de archivo PDF
uploaded_file = st.file_uploader("📄 Sube tu documento PDF", type=["pdf"])

if uploaded_file:
    # Procesamiento del archivo PDF
    temp_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(temp_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    all_splits = text_splitter.split_documents(docs)

    # Embeddings y vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    vector_store = Chroma(
        collection_name="pdf_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db"
    )
    vector_store.add_documents(documents=all_splits)

    st.success("✅ Documento cargado y procesado. Ya puedes hacer preguntas.")

    # LLM y prompt
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0.4,
        num_predict=258,
    )
    prompt = hub.pull("rlm/rag-prompt")

    class MyState(MessagesState):
        used_tool: NotRequired[bool]

    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """Retrieve information related to a query."""
        retrieved_docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    def query_or_respond(state: MyState):
        llm_with_tools = llm.bind_tools([retrieve])
        response = llm_with_tools.invoke(state["messages"])
        used_tool = bool(response.tool_calls and len(response.tool_calls) > 0)
        return {
            "messages": state["messages"] + [response],
            "used_tool": used_tool
        }

    tools_node = ToolNode([retrieve])

    def generate(state: MyState):
        recent_tool_messages = [m for m in state["messages"] if m.type == "tool"]
        docs_content = "\n\n".join(m.content for m in recent_tool_messages)

        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise.\n\n"
            f"{docs_content}"
        )

        conversation_messages = [
            m for m in state["messages"]
            if m.type in ("human", "system") or (m.type == "ai" and not m.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        response = llm.invoke(prompt)
        return {
            "messages": state["messages"] + [response],
            "used_tool": state.get("used_tool", False)
        }

    # Grafo
    graph_builder = StateGraph(MyState)
    graph_builder.add_node("query_or_respond", query_or_respond)
    graph_builder.add_node("tools", tools_node)
    graph_builder.add_node("generate", generate)
    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges("query_or_respond", tools_condition, {END: END, "tools": "tools"})
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    graph = graph_builder.compile()

    # Historial del chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("💬 Escribe tu pregunta sobre el documento...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        response = graph.invoke({"messages": [{"type": "human", "content": user_input}]})

        messages = response["messages"]
        final_message = messages[-1]
        assistant_response = final_message.content if hasattr(final_message, "content") else final_message["content"]

        with st.chat_message("assistant"):
            st.markdown(assistant_response)
            if response.get("used_tool"):
                st.markdown("⚙️ **Nota:** Se usó la herramienta de recuperación.")
            else:
                st.markdown("💡 **Nota:** No se usó la herramienta de recuperación.")

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
else:
    st.info("⬆️ Sube un archivo PDF para comenzar.")
