from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.summarize.chain import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from gtts import gTTS
import base64
import re
import pandas as pd
from PyPDF2 import PdfReader
from transformers import BlipProcessor, BlipForConditionalGeneration
# import torch
import streamlit as st
import io

# Defining max tokens
MAX_TOKENS = 6000


def chunk_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    return split_docs


def summarize_chain(docs, llm):
    total_text = " ".join(doc.page_content for doc in docs)
    total_tokens = len(total_text) // 4

    if total_tokens < MAX_TOKENS:
        prompt = PromptTemplate(input_variables=['text'], template=(
            '''Please provide a concise and detailed summary of the following content.
               Understand the type and message of the text provided.
               Add suitable title followed by introduction.
               Keep section-wise brief pointers (mentioning topics or highlights).
               End with a fitting conclusion.
               Text: {text}'''
        ))
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
        result = chain.invoke({"input_documents": docs})
        # If the output is a dict with metadata, get the actual summary text:
        if isinstance(result, dict):
            return result.get("output_text") or result.get("text") or str(result)
        return str(result)

    else:
        chunked_docs = chunk_documents(docs)
        initial_prompt = PromptTemplate(input_variables=['text'], template=(
            "You are an assistant for text summarization tasks. Write a concise and short summary of the provided text.\n{text}"
        ))
        final_prompt = PromptTemplate(input_variables=['text'], template=(
            '''Provide the final summary of the entire text with these important points.
               Add a suitable title. Start the precise summary with an introduction, state key notes in pointers and 
               end with conclusion.
               The provided text: {text}'''
        ))

        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            map_prompt=initial_prompt,
            combine_prompt=final_prompt
        )
        result = chain.invoke({"input_documents": chunked_docs})
        if isinstance(result, dict):
            return result.get("output_text") or result.get("text") or str(result)
        return str(result)


def extract_text_from_pdf(uploaded_file):

    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def process_csv_file(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df.to_string()  # Convert DataFrame to string for processing


def generate_audio(summary_text, lang="en"):
    # Formatting text for better audio
    text = re.sub(r'[#*_>`\-]', '', summary_text)
    text = re.sub(r'(?<=[^\.\!\?])\n', '. ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)

    # Use BytesIO to store the audio in memory instead of a file
    fp = io.BytesIO()
    tts = gTTS(text, lang=lang)
    tts.write_to_fp(fp)

    # Seek to the beginning of the buffer to read it
    fp.seek(0)
    audio_bytes = fp.read()

    b64 = base64.b64encode(audio_bytes).decode()
    return audio_bytes, b64

# Load BLIP model and processor


@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base")
    return processor, model


def confirm_deletion():
    """Legacy function - replaced by st.warning/columns in app.py chat clear."""
    pass
