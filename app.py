import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("Using pysqlite3 for enhanced sqlite support")
except ModuleNotFoundError:
    print("pysqlite3 not found — using built-in sqlite3")

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import streamlit.components.v1 as components
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
import base64
from dotenv import load_dotenv
import tempfile
import os
import validators
from YTutilities import get_transcript_as_document
from GenUtils import summarize_chain, generate_audio, confirm_deletion
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain_groq import ChatGroq
import streamlit as st

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
groq_api_key = os.getenv("GROQ_API_KEY")

# Page config
st.set_page_config(page_title="Skweezy AI", page_icon="🧠", layout="centered")

# App branding section
with st.container():
    st.markdown("""
        <style>
        .main-title {
            text-align: center;
            font-size: 3em;
            font-weight: bold;
            margin-bottom: 0;
        }
        .subtitle {
            text-align: center;
            color: #6c757d;
            margin-top: 0;
            margin-bottom: 40px;
        }
        .button-container {
            display: flex;
            justify-content: center;
            gap: 1.5em;
            flex-wrap: wrap;
        }
        .stButton>button {
            border-radius: 14px;
            padding: 12px 28px;
            font-size: 1.1em;
            font-weight: 500;
            background-color: #1f1f1f;
            border: 1px solid #3a3a3a;
            transition: all 0.2s ease-in-out;
            min-width: 140px;
        }
        .stButton>button:hover {
            background-color: #2c2c2c;
            border-color: #555;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="main-title">🧠 Skweezy AI</div>
        <div class="subtitle">Summarize YouTube, Websites, and PDFs on the fly — with downloadable audio!</div>
    """, unsafe_allow_html=True)

# Initialize LLM
llm = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=groq_api_key)

# Mode selection
if "mode" not in st.session_state:
    st.session_state.mode = None

# Custom button layout
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🎥 YouTube"):
        st.session_state.mode = "youtube"
with col2:
    if st.button("🌐 Website"):
        st.session_state.mode = "web"
with col3:
    if st.button("📄 PDF"):
        st.session_state.mode = "pdf"
with col4:
    if st.button("💬 Chat"):
        st.session_state.mode = "chat"

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown("---")

# Summary Logic
if st.session_state.mode == "youtube":
    url = st.text_input("Enter full YouTube video URL:")
    if url:
        if not validators.url(url) or "youtube.com" not in url:
            st.error("Please enter a valid YouTube URL.")
        else:
            try:
                with st.spinner("Summarizing video..."):
                    docs = get_transcript_as_document(url)
                    summary = summarize_chain(docs, llm)

                    audio_data, b64_audio = generate_audio(summary)
                    st.audio(audio_data, format="audio/mp3")
                    st.download_button("⬇️ Download Audio",
                                       data=audio_data, file_name="summary.mp3")

                    st.subheader("Summary")
                    st.success(summary)

                    b64_text = base64.b64encode(summary.encode()).decode()
                    st.markdown(
                        f'<a href="data:file/txt;base64,{b64_text}" download="summary.txt">📄 Download Summary</a>',
                        unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

elif st.session_state.mode == "web":
    url = st.text_input("Enter Website URL:")
    if url:
        if not validators.url(url) or "youtube.com" in url:
            st.error("Please enter a valid website URL (not YouTube).")
        else:
            try:
                with st.spinner("Summarizing website content..."):
                    loader = UnstructuredURLLoader(
                        urls=[url],
                        ssl_verify=True,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    docs = loader.load()

                    summary = summarize_chain(docs, llm)

                    audio_data, b64_audio = generate_audio(summary)
                    st.audio(audio_data, format="audio/mp3")
                    st.download_button("⬇️ Download Audio",
                                       data=audio_data, file_name="summary.mp3")

                    st.subheader("Summary")
                    st.success(summary)

                    b64_text = base64.b64encode(summary.encode()).decode()
                    st.markdown(
                        f'<a href="data:file/txt;base64,{b64_text}" download="summary.txt">📄 Download Summary</a>',
                        unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

elif st.session_state.mode == "pdf":
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        try:
            with st.spinner("Processing PDFs..."):
                docs = []
                for uploaded_file in uploaded_files:
                    temppdf = f"./temp_{uploaded_file.name}"
                    with open(temppdf, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    loader = PyPDFLoader(temppdf)
                    docs.extend(loader.load())
                    os.remove(temppdf)

                summary = summarize_chain(docs, llm)

                audio_data, b64_audio = generate_audio(summary)
                st.audio(audio_data, format="audio/mp3")
                st.download_button("⬇️ Download Audio",
                                   data=audio_data, file_name="summary.mp3")

                st.subheader("Summary")
                st.success(summary)

                b64_text = base64.b64encode(summary.encode()).decode()
                st.markdown(
                    f'<a href="data:file/txt;base64,{b64_text}" download="summary.txt">📄 Download Summary</a>',
                    unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")

elif st.session_state.mode == "chat":
    st.header("💬 Chat Mode")

    # KEEP the chat history logic
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask me anything…", key="chat_input")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            try:
                msg = HumanMessage(content=prompt)
                resp = llm.invoke([msg])
                answer = resp.content
                st.markdown(answer)
            except Exception as e:
                answer = f"❌ Error: {e}"
                st.error(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer})

    if st.button("🧹 Clear Chat"):
        st.session_state.show_confirmation = True

    if st.session_state.get('show_confirmation', False):
        st.warning("Are you sure you want to clear the chat history?")
        colA, colB = st.columns(2)
        with colA:
            if st.button("Yes, clear chat"):
                st.session_state.messages = []
                st.session_state.show_confirmation = False
                st.rerun()
        with colB:
            if st.button("No, keep chat"):
                st.session_state.show_confirmation = False
                st.rerun()
