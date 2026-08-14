# CHATBOT using OPENAI and OLLAMA

Streamlit Q&A chatbots built with LangChain:

- **1-OpenAI_ChatBot** — OpenAI models (API key entered in the sidebar)
- **2-Ollama_ChatBot** — Local Ollama models

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Optional: create a `.env` file with `LANGCHAIN_API_KEY` if you use LangSmith tracing.

## Run

**OpenAI chatbot**

```bash
streamlit run 1-OpenAI_ChatBot/app.py
```

**Ollama chatbot** (Ollama must be running locally with the selected model pulled)

```bash
streamlit run 2-Ollama_ChatBot/main.py
```
