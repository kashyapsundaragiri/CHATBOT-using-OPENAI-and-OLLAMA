import os
import re
import time

import arxiv
import streamlit as st
import wikipedia
from dotenv import load_dotenv
from langchain_classic.agents import AgentType, initialize_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()

wikipedia.set_user_agent("QandA-ChatBot/1.0 (educational; LangChain learner)")


class WikiInput(BaseModel):
    query: str = Field(description="query to look up on wikipedia")


class ArxivInput(BaseModel):
    query: str = Field(
        description="ArXiv search query or paper id, e.g. '1706.03762' or 'attention is all you need'"
    )


def safe_wikipedia(query: str) -> str:
    last_error = None
    for attempt in range(3):
        try:
            titles = wikipedia.search(query, results=1)
            if not titles:
                return "No good Wikipedia Search Result was found"
            summary = wikipedia.summary(titles[0], sentences=4, auto_suggest=False)
            return f"Page: {titles[0]}\nSummary: {summary[:300]}"
        except wikipedia.exceptions.DisambiguationError as e:
            try:
                summary = wikipedia.summary(e.options[0], sentences=4, auto_suggest=False)
                return f"Page: {e.options[0]}\nSummary: {summary[:300]}"
            except Exception as inner:
                last_error = inner
        except Exception as e:
            last_error = e
            time.sleep(1.5 * (attempt + 1))
    return f"Wikipedia lookup failed after retries: {type(last_error).__name__}: {last_error}"


def safe_arxiv(query: str) -> str:
    try:
        client = arxiv.Client()
        cleaned = query.strip()
        if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", cleaned):
            search = arxiv.Search(id_list=[cleaned.split("v")[0]], max_results=1)
        else:
            search = arxiv.Search(
                query=cleaned, max_results=1, sort_by=arxiv.SortCriterion.Relevance
            )
        results = list(client.results(search))
        if not results:
            return "No good ArXiv Search Result was found"
        paper = results[0]
        summary = (paper.summary or "").replace("\n", " ")[:300]
        authors = ", ".join(a.name for a in paper.authors[:5])
        return (
            f"Title: {paper.title}\n"
            f"Authors: {authors}\n"
            f"Published: {paper.published.date() if paper.published else 'N/A'}\n"
            f"Entry ID: {paper.entry_id}\n"
            f"Summary: {summary}"
        )
    except Exception as e:
        return f"ArXiv lookup failed: {type(e).__name__}: {e}"


wikipedia_tool = StructuredTool.from_function(
    func=safe_wikipedia,
    name="wikipedia",
    description=(
        "A wrapper around Wikipedia. Useful for general questions about people, "
        "places, companies, historical events, or other subjects. Input should be a search query."
    ),
    args_schema=WikiInput,
)

arxiv_tool = StructuredTool.from_function(
    func=safe_arxiv,
    name="arxiv",
    description=(
        "A wrapper around Arxiv.org for scientific papers. Input should be a search "
        "query or an arXiv paper id like 1706.03762."
    ),
    args_schema=ArxivInput,
)

search = DuckDuckGoSearchRun(
    name="search",
    description="Search the web for the latest information on a given topic",
)

st.title("🔎 LangChain - Chat with search")
st.markdown(
    """
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""
)

st.sidebar.title("Search Engine Settings")
api_key_input = st.sidebar.text_input(
    "Enter your Groq API key",
    type="password",
    help="Leave blank to use GROQ_API_KEY from your .env file.",
)
model_name = st.sidebar.selectbox(
    "Groq model",
    ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.6-27b", "groq/compound-mini"],
    index=0,
)

api_key = (api_key_input or os.getenv("GROQ_API_KEY") or "").strip()

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not api_key:
        st.warning("Enter a Groq API key in the sidebar, or set GROQ_API_KEY in .env.")
        st.stop()

    if not api_key.startswith("gsk_"):
        st.error("That doesn't look like a Groq key (should start with `gsk_`).")
        st.stop()

    llm = ChatGroq(model=model_name, api_key=api_key, streaming=True)
    tools = [arxiv_tool, wikipedia_tool, search]
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
    )

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(
            st.container(), expand_new_thoughts=True
        )
        response = agent.run(prompt, callbacks=[streamlit_callback])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
