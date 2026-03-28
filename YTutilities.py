import os
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
from langchain_core.documents import Document
import yt_dlp
import streamlit as st

load_dotenv()


def extract_youtube_video_id(url: str) -> str:
    """Extract the YouTube video ID from a full URL (long or short form)."""
    if "youtube.com" in url:
        return parse_qs(urlparse(url).query).get('v', [None])[0]
    elif "youtu.be" in url:
        return url.split("/")[-1]
    return None


def get_transcript_as_document(url: str, languages=None, preserve_formatting=False) -> list[Document]:
    video_id = extract_youtube_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL: could not extract video id")

    try:
        st.info("Getting video metadata and description...")
        # Always reliable: metadata + description for summary
        ydl_opts = {'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        title = info.get('title', 'Unknown')
        description = info.get('description', '')[:4000]
        channel = info.get('uploader', 'Unknown')
        duration = info.get('duration', 0)

        full_text = f"""Title: {title}
Channel: {channel}
Duration: {duration}s

Description:
{description}

This video content summary is based on available metadata and description as full transcript extraction is blocked by YouTube."""

        return [Document(page_content=full_text)]

    except Exception as e:
        raise RuntimeError(
            f"Video processing failed for video_id={video_id}. Exception: {e}")
