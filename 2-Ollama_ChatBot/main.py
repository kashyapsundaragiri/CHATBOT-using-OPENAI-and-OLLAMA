from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st

import os
from dotenv import load_dotenv

load_dotenv()


# Langsmith tracking

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = " Ollama Q&A ChatBot"

# Prompt Template

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user query in a concise and helpful manner. \n\n"),
        ("user", "Question: {question}"),
    ]
)

def get_response(question, engine, temperature, max_tokens):
    llm = Ollama(model=engine)
    output_parser = StrOutputParser()
    chain = prompt_template | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer


# Streamlit App

st.title("Ollama Q&A ChatBot")

# Dropdown for model selection

llm = st.sidebar.selectbox("Select a model", ["gemma2", "phi3", "mistral", "deepseek-r1"])

# Adjust response parameters

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
max_tokens = st.sidebar.slider("Max Tokens", min_value=100, max_value=1000, value=500, step=100)

# Main interface for user input

st.write("Ask me anything about the topic you want to know more about")

user_input = st.text_input("You:")

if user_input:
    with st.spinner("Thinking..."):
        response = get_response(user_input, llm, temperature, max_tokens)
        st.write("Bot:", response)
else:
    st.write("Please enter a question to get started")
    
    

