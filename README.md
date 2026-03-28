# Skweezy AI 🧠

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Groq](https://img.shields.io/badge/Groq-00D2FF?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com)

Skweezy AI is a powerful Streamlit web application that provides AI-powered summarization for YouTube videos, websites, PDFs, and other files. Generate concise summaries with downloadable audio (MP3) and text files. Features a chat mode for Q&A with document context using Groq's fast LLM inference.

## ✨ Features
- **🎥 YouTube Summarizer**: Extract metadata & description, generate AI summary + audio.
- **🌐 Website Summarizer**: Scrape & summarize web content.
- **📄 PDF Summarizer**: Multi-PDF upload, extract text, summarize.
- **💬 Chat Mode**: Conversational AI with optional file context (PDF/txt/csv).
- **🎵 Audio Export**: Download summaries as MP3 via gTTS.
- **📥 Text Download**: Summary as .txt.
- **📱 Responsive UI**: Adaptive light/dark theme matching system preferences.
- **Fast Inference**: Powered by Groq (gpt-oss-120b model).

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [Groq API Key](https://console.groq.com/keys) (free tier available)
- Optional: [LangChain API Key](https://smith.langchain.com/) for tracing

### Setup
1. Clone/download the repo:
   ```
   git clone <repo-url>
   cd Skweezy_AI
   ```

2. Install dependencies (using uv recommended):
   ```
   uv sync  # or pip install -r requirements.txt
   ```

3. Create `.env` file:
   ```
   GROQ_API_KEY=your_groq_key_here
   LANGCHAIN_API_KEY=your_langchain_key_here  # Optional
   ```

4. Run the app:
   ```
   streamlit run app.py
   ```
   Opens at http://localhost:8501

## 📖 Usage

1. **Select Mode** via top buttons:
   | Mode | Input | Output |
   |------|--------|--------|
   | 🎥 YouTube | Paste URL | Summary + audio/text download |
   | 🌐 Website | Paste URL | Summary + audio/text |
   | 📄 PDF | Upload files | Combined summary + audio/text |
   | 💬 Chat | Type query (+ optional file) | Conversational responses |

2. **Downloads**: Each summary provides:
   - ▶️ Play audio inline
   - ⬇️ Download MP3 (`summary.mp3`)
   - 📄 Download TXT (`summary.txt`)

**Note**: YouTube uses video description/metadata (full transcripts blocked by YouTube policy).

## 🛠 Tech Stack
- **Frontend**: Streamlit (adaptive theme)
- **Backend**: LangChain (summarization chains), Groq LLM
- **Processing**: yt-dlp (YouTube), WebBaseLoader (web), PyPDF2 (PDF), gTTS (audio), pandas (CSV)
- **Project Mgmt**: uv / pyproject.toml

## 🔧 Customization
- Edit `app.py` for UI changes.
- Modify `GenUtils.py` for summarization prompts.
- Update `YTutilities.py` for YouTube logic.

## 🤖 API Keys
- **GROQ_API_KEY** (required): Sign up at [console.groq.com](https://console.groq.com)
- **LANGCHAIN_API_KEY** (optional): For LangSmith tracing.

## ❗ Troubleshooting
- **No summary?** Check API key, internet.
- **YouTube fails?** Use public videos; private/unlisted not supported.
- **Web scrape fails?** Some sites block bots (app uses UA header).
- **Theme issues?** Toggle system light/dark (macOS: System Settings > Appearance).
- **Deps errors?** `uv sync --dev` or `pip install -r requirements.txt --upgrade`.

## 📈 Performance
- **Summary Quality**: Map-reduce for long docs (>6k tokens).
- **Speed**: Groq ~100+ tokens/sec.
- **Limits**: Respects LLM context; chunks large inputs.

## 🤝 Contributing
1. Fork & PR.
2. Add features (e.g., full RAG with Chroma).
3. Test locally.
4. Update README.

## 📄 License
MIT License - feel free to use/modify.

## 🙏 Acknowledgments
Built with [Streamlit](https://streamlit.io), [LangChain](https://langchain.com), [Groq](https://groq.com).

