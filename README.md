# Q&A Chatbots

Interactive Q&A chatbot apps built with **LangChain** and **Streamlit**.

| App | Backend | Where it runs |
| --- | --- | --- |
| OpenAI ChatBot | OpenAI API (`gpt-4o`, `gpt-4-turbo`, `gpt-4`) | Local or [Streamlit Community Cloud](https://streamlit.io/cloud) |
| Ollama ChatBot | Local Ollama models (`gemma2`, `phi3`, `mistral`, `deepseek-r1`) | Local only (requires [Ollama](https://ollama.com)) |
| RAG Document Q&A | Groq + OpenAI embeddings + FAISS over research PDFs | Local |
| Chat With PDF | Groq + HuggingFace embeddings + Chroma with chat history | Local |

---

## Demo

### Interface

![Q&A ChatBot Interface](IMAGES/Interface.png)

### Input

![User Input](IMAGES/Input.png)

### Output

![Bot Output](IMAGES/Output.png)

---

## Project structure

```text
Q&A Chatbots/
├── 1-OpenAI_ChatBot/
│   └── app.py
├── 2-Ollama_ChatBot/
│   └── main.py
├── 3-RAG_Document_Q&A/
│   ├── app.py
│   └── research_papers/
├── 4- Chat_With_PDF/
│   └── app.py
├── IMAGES/
│   ├── Interface.png
│   ├── Input.png
│   └── Output.png
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Features

- Clean Streamlit UIs for asking questions and viewing answers
- OpenAI and local Ollama chatbot apps
- RAG over PDF research papers (FAISS)
- Multi-PDF chat with conversational history (Chroma)
- Optional LangSmith tracing via environment variables
- API keys entered in the UI or loaded from local `.env` (never committed)

---

## Prerequisites

- Python 3.10+
- For OpenAI / RAG embeddings: an [OpenAI API key](https://platform.openai.com/api-keys)
- For Groq apps: a [Groq API key](https://console.groq.com/)
- For Ollama app: [Ollama](https://ollama.com) installed and running

---

## Setup

1. **Clone the repository**

```bash
git clone https://github.com/kashyapsundaragiri/QA-Chatbots.git
cd QA-Chatbots
```

2. **Create and activate a virtual environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **(Optional) Configure environment variables**

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Edit `.env` with your own keys. **Do not commit real keys.**

```env
LANGCHAIN_API_KEY=your_langsmith_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

---

## Run locally

Always prefer the venv Streamlit binary so packages resolve correctly:

```bash
# Windows
venv\Scripts\streamlit.exe run 1-OpenAI_ChatBot/app.py
venv\Scripts\streamlit.exe run 2-Ollama_ChatBot/main.py
venv\Scripts\streamlit.exe run "3-RAG_Document_Q&A/app.py"
venv\Scripts\streamlit.exe run "4- Chat_With_PDF/app.py"
```

---

## Deploy (OpenAI app only)

1. Push your code to GitHub
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/)
3. Deploy with:
   - **Main file path:** `1-OpenAI_ChatBot/app.py`
   - **Python dependencies:** `requirements.txt`

**Never** put API keys in the README, screenshots, or committed files.

---

## Security

- `.env` is ignored by Git — keep real credentials there locally only
- Use `.env.example` as a safe template for collaborators
- Rotate any key that may have been exposed

---

## Tech stack

- [Streamlit](https://streamlit.io/) — UI
- [LangChain](https://www.langchain.com/) — prompt + chain orchestration
- [OpenAI](https://openai.com/) — cloud LLMs / embeddings
- [Groq](https://groq.com/) — fast LLM inference
- [Ollama](https://ollama.com/) — local LLMs
- FAISS / Chroma — vector stores

---

## License

This project is licensed under the [MIT License](LICENSE).
