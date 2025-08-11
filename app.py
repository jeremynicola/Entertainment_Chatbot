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

# === LIVE INTERNET SEARCH (DuckDuckGo + fallback parsing) ===
def live_search(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=1))

        if not results:
            st.warning("❌ No search results found.")
            return None

        top_link = results[0].get("href")
        st.info(f"🔍 Found link: {top_link}")

        # Try Newspaper3k
        content = None
        try:
            article = Article(top_link)
            article.download()
            article.parse()
            content = article.text.strip()
        except Exception as e:
            st.warning(f"⚠ Newspaper3k failed: {e}")

        # Fallback to Trafilatura
        if not content:
            try:
                downloaded = trafilatura.fetch_url(top_link)
                if downloaded:
                    content = trafilatura.extract(downloaded)
            except Exception as e:
                st.warning(f"⚠ Trafilatura failed: {e}")

        return content[:2000] if content else None

    except Exception as e:
        st.error(f"Live search error: {type(e).__name__} - {e}")
        return None

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

    # 2. If no answer, try live search
    if not answer or answer.strip().lower() in ["", "i don't know", "not sure"]:
        with st.spinner("🌍 Searching the web..."):
            search_result = live_search(user_input)
            if search_result:
                conversation_history = ''.join([f"{m['sender']}: {m['message']}\n" for m in st.session_state.chat_history])
                rewrite_prompt = f"""
You are an entertainment expert.
Rewrite the following extracted web content into a friendly, clear answer:

User question: {user_input}
Web content: {search_result}

Conversation so far:
{conversation_history}
"""
                answer = llm.invoke(rewrite_prompt).content
            else:
                answer = "Sorry, I couldn't find an answer in the documents or online."

    st.session_state.chat_history.append({"sender": "bot", "message": answer})
    st.session_state.memory.chat_memory.add_ai_message(answer)

# === Render chat history ===
for chat in st.session_state.chat_history:
    render_message(chat["message"], chat["sender"])
