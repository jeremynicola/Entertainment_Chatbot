import os
import streamlit as st
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS

# 🔹 Extra imports
from duckduckgo_search import DDGS
from newspaper import Article
import trafilatura


# === SETTINGS ===
DATA_DIR = "./data"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PRIMARY_MODEL = "llama-3.3-70b-versatile"     # preferred
FALLBACK_MODEL = "llama-3.1-8b-instant"       # fallback if primary fails


# === INIT LLM WITH FALLBACK ===
@st.cache_resource
def initialize_llm():
    groq_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not groq_key:
        st.error("❌ Missing GROQ_API_KEY. Add it in `.streamlit/secrets.toml` or environment.")
        st.stop()

    try:
        return ChatGroq(
            temperature=0.5,
            api_key=groq_key,          # ✅ FIX: correct param
            model_name=PRIMARY_MODEL,
        )
    except Exception as e:
        st.warning(f"⚠ Primary model {PRIMARY_MODEL} failed: {e}. Falling back to {FALLBACK_MODEL}...")
        return ChatGroq(
            temperature=0.5,
            api_key=groq_key,          # ✅ FIX
            model_name=FALLBACK_MODEL,
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
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )


# === LIVE INTERNET SEARCH ===
def live_search(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=1))

        if not results:
            st.warning("❌ No search results found.")
            return None

        top_link = results[0].get("href")
        st.info(f"🔍 Found link: {top_link}")

        content = None
        try:
            article = Article(top_link)
            article.download()
            article.parse()
            content = article.text.strip()
        except Exception as e:
            st.warning(f"⚠ Newspaper3k failed: {e}")

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


# === BAD ANSWER DETECTION ===
BAD_ANSWERS = ["i don't know", "not sure", "sorry", "cannot find", "no information"]

def is_bad_answer(ans: str) -> bool:
    if not ans:
        return True
    ans_low = ans.lower()
    if any(bad in ans_low for bad in BAD_ANSWERS):
        return True
    # too short or vague
    if len(ans.split()) < 15:
        return True
    return False


# === STYLE MESSAGE RENDERER ===
def render_message(message, sender):
    timestamp = datetime.now().strftime("%H:%M")
    color = "#007BFF" if sender == "user" else "#000000"
    align = "margin-left:auto;" if sender == "user" else "margin-right:auto;"
    st.markdown(
        f"""
        <div style="background-color:{color}; padding:10px; border-radius:10px; 
             max-width:70%; {align} margin-bottom:5px; border:1px solid #ccc; color:white;">
            {message}
            <div style="font-size:10px; text-align:right; color:lightgray;">{timestamp}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# === MAIN APP ===
st.title("🎬 Entertainment Chatbot")
st.caption("Ask about movies, games, or celebrities!")

if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    

user_input = st.text_input("Type your message...")

if user_input:
    llm = initialize_llm()
    db = load_or_create_db()
    answer = None

    # Save user message
    st.session_state.chat_history.append({"sender": "user", "message": user_input})
    st.session_state.memory.chat_memory.add_user_message(user_input)

    # 1. Try database
    if db:
        qa_chain = setup_chain(llm, db)
        with st.spinner("🎬 Thinking with database..."):
            try:
                result = qa_chain.invoke({"query": user_input})
                answer = result.get("result", "").strip()
            except Exception as e:
                st.warning(f"⚠ Database query failed: {e}")

    # 2. If bad/empty answer → do web search
    if is_bad_answer(answer):
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
                try:
                    response = llm.invoke(rewrite_prompt)
                    answer = getattr(response, "content", str(response))
                except Exception as e:
                    st.error(f"Groq rewrite failed: {e}")
                    answer = "Sorry, I couldn't process the web result."
            else:
                answer = "Sorry, I couldn't find an answer in the documents or online."

    # Save bot message
    st.session_state.chat_history.append({"sender": "bot", "message": answer})
    st.session_state.memory.chat_memory.add_ai_message(answer)

# === Render chat history ===
for chat in st.session_state.chat_history:
    render_message(chat["message"], chat["sender"])
