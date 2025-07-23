import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# import openai
# from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]= os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]= "Simple Q&A Chatbot With Ollama"
os.environ["LANGCHAIN_TRACING_V2"]= "true"

## Prompt Template
prompt= ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful Assistant. Please repsonse to the user Queries."),
        ("user", "question: {question}")
    ]
)

from langchain_community.llms import Ollama
def generate_response(question, engine, temperature, max_tokens):

    llm= Ollama(model= engine,
                base_url="http://localhost:11434",
                temperature= temperature
                )

    output_parser= StrOutputParser()
    chain= prompt|llm|output_parser
    
    answer= chain.invoke({'question': question})
    return answer

## Title of the app
st.title("Enhanced Q&A Chatbot With Ollama")

## Sidebar-- Select the OpenAI model
engine= st.sidebar.selectbox("Select Open Source model", ["phi", "phi3:mini", "gemma3:1b"])
#ollama run phi
#ollama run phi3:mini
#ollama run gemma3:1b 


## Sidebar-- Adjust response parameter
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## Main interface for user input
st.write("Go ahead and ask any question")
user_question= st.text_area("You:")


# This code is a simple Q&A chatbot application built using Streamlit and LangChain.
# It allows users to input questions and receive answers from an open-source model like Mistral
# Llama3, or Llama2, with adjustable parameters for temperature and max tokens.

if user_question:    
    response = generate_response(user_question, engine, temperature, max_tokens)    
    st.write(response)
    st.markdown("**Bot:**")
    st.success(response)


else:
    st.write("Please enter your question.")