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
from GenUtils import summarize_chain, generate_audio, confirm_deletion, extract_text_from_pdf, process_csv_file, describe_image
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
        .stButton > button {
            border-radius: 14px !important;
            padding: 12px 28px !important;
            font-size: 1.1em !important;
            font-weight: 500 !important;
            background: var(--p1-50, linear-gradient(var(--bg-base-bg, white), var(--bg-base-bg, white))) !important;
            color: var(--text-primary, black) !important;
            border: 1px solid var(--border-color, #ddd) !important;
            transition: all 0.2s ease-in-out !important;
            min-width: 140px !important;
            box-shadow: var(--shadow-md, 0 1px 2px 0 rgb(0 0 0 / 0.05)) !important;
        }
        .stButton > button:hover {
            background: var(--p1-100, linear-gradient(var(--bg-base-bg, #f0f0f0), var(--bg-base-bg, #f0f0f0))) !important;
            border-color: var(--p1-200, #999) !important;
            color: var(--text-primary, black) !important;
            transform: translateY(-1px) !important;
        }
        .stButton > button:active {
            transform: translateY(0) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="main-title">🧠 Skweezy AI</div>
        <div class="subtitle">Summarize YouTube, Websites, and PDFs on the fly — with downloadable audio!</div>
    """, unsafe_allow_html=True)

# Initialize LLM
llm = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=groq_api_key)
vision_llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    groq_api_key=groq_api_key
)
# Mode selection
if "mode" not in st.session_state:
    st.session_state.mode = None

# Custom button layout
col1, col2, col3, col4, col5 = st.columns(5)

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

with col5:
    if st.button("🖼️ Image"):
        st.session_state.mode = "image"

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
                    # Use WebBaseLoader instead of Unstructured
                    from langchain_community.document_loaders import WebBaseLoader

                    loader = WebBaseLoader(url)
                    # Add headers to bypass basic bot detection
                    loader.requests_kwargs = {
                        "headers": {"User-Agent": "Mozilla/5.0"}}

                    docs = loader.load()

                    if not docs or not docs[0].page_content.strip():
                        st.error(
                            "Could not extract any text from this website.")
                    else:
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

elif st.session_state.mode == "image":
    uploaded_image = st.file_uploader(
        "Upload an image (png, jpg, jpeg)", type=['png', 'jpg', 'jpeg'])
    if uploaded_image:
        image_bytes = uploaded_image.read()
        st.image(image_bytes, caption="Uploaded Image", width=250)
        with st.spinner("Generating precise description..."):
            try:
                description = describe_image(image_bytes, vision_llm)
                audio_data = generate_audio(description)[0]
                st.audio(audio_data, format="audio/mp3")
                st.download_button("⬇️ Download Audio", audio_data,
                                   file_name="image_description.mp3")
                st.subheader("🖼️ Precise Description")
                st.success(description)
                b64_text = base64.b64encode(description.encode()).decode()
                st.markdown(
                    f'<a href="data:file/txt;base64,{b64_text}" download="description.txt">📄 Download Text</a>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

elif st.session_state.mode == "file_upload":
    uploaded_file = st.file_uploader("Upload a file (e.g., .txt, .pdf, .csv):")
    if uploaded_file is not None:
        try:
            with st.spinner("Processing file..."):
                # Determine file type and process accordingly
                file_type = uploaded_file.name.split('.')[-1].lower()

                if file_type in ['txt']:
                    content = uploaded_file.read().decode("utf-8")
                elif file_type in ['pdf']:
                    content = extract_text_from_pdf(uploaded_file)
                elif file_type in ['csv']:
                    content = process_csv_file(uploaded_file)
                else:
                    st.error(
                        "Unsupported file type. Please upload a .txt, .pdf, or .csv file.")
                    content = None

                if content:
                    # Generate insights or summary
                    docs = process_content_as_document(content)
                    summary = summarize_chain(docs, llm)

                    audio_data, b64_audio = generate_audio(summary)
                    st.audio(audio_data, format="audio/mp3")
                    st.download_button("⬇️ Download Audio",
                                       data=audio_data, file_name="summary.mp3")
                    st.subheader("Summary")
                    st.write(summary)
        except Exception as e:
            st.error(f"An error occurred: {e}")

elif st.session_state.mode == "chat":
    st.header("💬 Chat Mode")

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # File upload for context
    uploaded_file = st.file_uploader(
        "Upload a file to provide context (e.g., .txt, .pdf, .csv):")
    file_context = None
    if uploaded_file:
        try:
            with st.spinner("Processing file..."):
                file_type = uploaded_file.name.split('.')[-1].lower()

                if file_type == 'txt':
                    file_context = uploaded_file.read().decode("utf-8")
                elif file_type == 'pdf':
                    file_context = extract_text_from_pdf(uploaded_file)
                elif file_type == 'csv':
                    file_context = process_csv_file(uploaded_file)
                else:
                    st.error(
                        "Unsupported file type. Please upload a .txt, .pdf, or .csv file.")
        except Exception as e:
            st.error(f"Error processing file: {e}")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input("Ask me anything…", key="chat_input")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process the prompt with file context
        with st.chat_message("assistant"):
            try:
                if file_context:
                    # Combine file context with user prompt
                    combined_prompt = f"Context:\n{file_context}\n\nQuestion:\n{prompt}"
                else:
                    combined_prompt = prompt

                msg = HumanMessage(content=combined_prompt)
                resp = llm.invoke([msg])
                answer = resp.content
                st.markdown(answer)
            except Exception as e:
                answer = f"❌ Error: {e}"
                st.error(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer})

        # Auto-scroll to bottom
        st.rerun()

    # Clear chat button
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
