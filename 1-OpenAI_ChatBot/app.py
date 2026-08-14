import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate




import os
from dotenv import load_dotenv

load_dotenv()


# Langsmith tracking (optional — skip if key is not set)
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "OpenAi Q&A ChatBot"


# Prompt Template

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user query in a concise and helpful manner. \n\n"),
        ("user", "Question: {question}"),
    ]
)


def get_response(question, api_key, llm, temperature, max_tokens):
    llm = ChatOpenAI(
        model=llm,
        api_key=api_key.strip(),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    output_parser = StrOutputParser()
    chain = prompt_template | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer


# Streamlit App

st.title("OpenAi Q&A ChatBot")

# Sidebar for settings

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAi API Key", type="password")

# Dropdown for model selection

llm = st.sidebar.selectbox("Select a model", ["gpt-4o", "gpt-4-turbo", "gpt-4"])

# Adjust response parameters

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
max_tokens = st.sidebar.slider("Max Tokens", min_value=100, max_value=1000, value=500, step=100)

# Main interface for user input

st.write("Ask me anything about the topic you want to know more about")

user_input = st.text_input("You:")

if user_input:
    if not api_key or not api_key.strip():
        st.warning("Please enter your OpenAI API key in the sidebar.")
    else:
        with st.spinner("Thinking..."):
            try:
                response = get_response(user_input, api_key, llm, temperature, max_tokens)
                st.write("Bot:", response)
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.write("Please enter a question to get started")



