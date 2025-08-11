import os
import streamlit as st
from datetime import datetime
from langchain_groq import ChatGroq
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS


# === SETTINGS ===
DATA_DIR = "./data"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "llama3-70b-8192"

# === INIT LLM ===
@st.cache_resource
def initialize_llm():
    return ChatGroq(
        temperature=0.5,
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model_name=MODEL_NAME
    )

# === BUILD / LOAD VECTOR DB ===
@st.cache_resource
def load_or_create_db():
    if not os.path.exists(DATA_DIR):
        return None
    pdfs = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    if len(pdfs) == 0:
        return None
    loader = DirectoryLoader(DATA_DIR, glob='*.pdf', loader_cls=PyPDFLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = FAISS.from_documents(texts, embeddings)
    return db

# === MEMORY ===
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === SETUP RETRIEVAL CHAIN ===
def setup_chain(llm, vector_db):
    retriever = vector_db.as_retriever()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )


# === STYLE MESSAGE RENDERER ===
def render_message(message, sender):
    timestamp = datetime.now().strftime("%H:%M")
    color = "Blue" if sender == "user" else "Black"
    align = "margin-left:auto;" if sender == "user" else "margin-right:auto;"
    st.markdown(
        f"""
        <div style="background-color:{color}; padding:10px; border-radius:10px; max-width:70%; {align} margin-bottom:5px; border:1px solid #ccc;">
            {message}
            <div style="font-size:10px; text-align:right; color:gray;">{timestamp}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# === MAIN APP ===
st.title("🎬 Entertainment Chatbot")
st.caption("Ask about movies, games, or celebrities!")

user_input = st.text_input("Type your message...")

# Clear chat history button
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.experimental_rerun()

user_input = st.text_input("Type your message...", key="user_message")

if user_input:
    llm = initialize_llm()
    db = load_or_create_db()
    answer = None

    st.session_state.chat_history.append({"sender": "user", "message": user_input})
    st.session_state.memory.chat_memory.add_user_message(user_input)

    # 1. Try database
    if db:
        qa_chain = setup_chain(llm, db)
        with st.spinner("🎬 Thinking with database..."):
            result = qa_chain.invoke({"query": user_input})
            answer = result["result"]

   
            else:
                answer = "Sorry, I couldn't find an answer in the databases."

    
    # Save conversation
    st.session_state.chat_history.append({"sender": "user", "message": user_input})
    st.session_state.memory.chat_memory.add_user_message(user_input)
    st.session_state.chat_history.append({"sender": "bot", "message": answer})
    st.session_state.memory.chat_memory.add_ai_message(answer)

    # Clear input
    st.session_state.user_message = ""
    st.experimental_rerun()


# === Render chat history ===
for chat in st.session_state.chat_history:
    render_message(chat["message"], chat["sender"])
