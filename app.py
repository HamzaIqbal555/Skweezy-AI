import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from GenUtils import summarize_chain, generate_audio, load_blip_model
from YTutilities import get_transcript_as_document
import validators
import os
from dotenv import load_dotenv
import base64
from PIL import Image
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.stylable_container import stylable_container
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

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
            margin-bottom: 30px;
        }
        .stButton>button {
            border-radius: 12px;
            height: 3em;
            width: 100%;
            font-size: 1.1em;
        }
        .mode-container {
            display: flex;
            justify-content: center;
            gap: 2em;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="main-title">🧠 Skweezy AI</div>
        <div class="subtitle">Summarize YouTube, Websites, and PDFs on the fly — with downloadable audio! Or Get an image Description</div>
    """, unsafe_allow_html=True)

# Initialize LLM
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

# Mode selection
if "mode" not in st.session_state:
    st.session_state.mode = None

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
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
    if st.button("🖼️ Image"):
        st.session_state.mode = "image"

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
                        f'<a href="data:file/txt;base64,{b64_text}" download="summary.txt">📄 Download Summary</a>', unsafe_allow_html=True)
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
                        f'<a href="data:file/txt;base64,{b64_text}" download="summary.txt">📄 Download Summary</a>', unsafe_allow_html=True)
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
                    f'<a href="data:file/txt;base64,{b64_text}" download="summary.txt">📄 Download Summary</a>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")

elif st.session_state.mode == "image":
    image_file = st.file_uploader(
        "Upload an image (JPG, PNG, etc.)", type=["png", "jpg", "jpeg"])
    if image_file:
        try:
            image = Image.open(image_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
            processor, model = load_blip_model()
            with st.spinner("Analyzing and describing the image..."):
                inputs = processor(images=image, return_tensors="pt")
                out = model.generate(**inputs)
                caption = processor.decode(out[0], skip_special_tokens=True)

                st.success("**Description:** " + caption)

                # Generate audio of description
                audio_data, b64 = generate_audio(caption)
                st.audio(audio_data, format="audio/mp3")
                st.markdown(
                    f'<div style="text-align: right;"><a href="data:audio/mp3;base64,{b64}" download="image_description.mp3">Download Audio</a></div>',
                    unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")
