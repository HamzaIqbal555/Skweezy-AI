# YTutilities.py

import os
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
# If you still need proxies via Webshare, keep this; otherwise you can remove.
from youtube_transcript_api.proxies import WebshareProxyConfig

# Adjust for your version of langchain
from langchain_core.documents import Document

load_dotenv()


def extract_youtube_video_id(url: str) -> str:
    """Extract the YouTube video ID from a full URL (long or short form)."""
    if "youtube.com" in url:
        return parse_qs(urlparse(url).query).get('v', [None])[0]
    elif "youtu.be" in url:
        return url.split("/")[-1]
    return None


def get_transcript_as_document(url: str, languages=None, preserve_formatting=False) -> list[Document]:
    """
    Fetch the transcript of the YouTube video and wrap it in a list of Document objects.
    :param url: full YouTube video URL
    :param languages: optional list of language codes (e.g., ['en'])
    :param preserve_formatting: whether to preserve HTML formatting tags
    :return: list of Documents (one per transcript chunk or a single one)
    :raises RuntimeError if transcript cannot be fetched
    """
    video_id = extract_youtube_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL: could not extract video id")

    try:
        # Setup optional proxy config if you use one
        proxy_cfg = WebshareProxyConfig(
            proxy_username=os.getenv("proxy_username"),
            proxy_password=os.getenv("proxy_password"),
        ) if (os.getenv("proxy_username") and os.getenv("proxy_password")) else None

        # Call correct static method (per current library design) to fetch transcript
        ytt = YouTubeTranscriptApi()
        transcript_data = ytt.fetch(
            video_id,
            languages=languages or ['en'],
            preserve_formatting=preserve_formatting,
            # If using proxies: use proxies parameter, not proxy_config
            # e.g., proxies={"http": "...", "https": "..."}
        )

        # Combine transcript texts into one continuous text block (or adapt to chunks)
        full_text = "\n".join(
            snippet.text for snippet in transcript_data.snippets)

        return [Document(page_content=full_text)]

    except Exception as e:
        # Provide clear error context
        raise RuntimeError(
            f"Transcript fetch failed for video_id={video_id}. Exception: {e}")
