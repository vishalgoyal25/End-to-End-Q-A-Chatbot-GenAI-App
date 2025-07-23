import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]= os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]= "Simple Q&A Chatbot With OPENAI"
os.environ["LANGCHAIN_TRACING_V2"]= "true"

## Prompt Template
prompt= ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful Assistant. Please repsonse to the user Queries."),
        ("user", "question: {question}")
    ]
)

def generate_response(question, api_key, engine, temperature, max_tokens):

    openai.api_key= api_key
    llm= ChatOpenAI(model= engine)

    output_parser= StrOutputParser()
    chain= prompt|llm|output_parser
    
    answer= chain.invoke({'question': question})
    return answer

## Title of the app
st.title("Enhanced Q&A Chatbot With OpenAI")

## Sidebar-- for settings
st.sidebar.title("Settings")
api_key= st.sidebar.text_input("Enter your Open AI API Key:", type="password")

## Sidebar-- Select the OpenAI model
engine= st.sidebar.selectbox("Select Open AI model",["gpt-4o","gpt-4-turbo","gpt-4"])

## Sidebar-- Adjust response parameter
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## Main interface for user input
st.write("Go ahead and ask any question")
user_question= st.text_input("You:")

if user_question and api_key:
    response= generate_response(user_question, api_key, engine, temperature, max_tokens)
    st.write(response)

elif user_question:
    st.warning("Please enter the OpenAI API_KEY in the Side Bar")
else:
    st.write("Please enter the user input")


# This code is a simple Q&A chatbot application built using Streamlit and OpenAI's API.

# It allows users to input questions and receive answers from the OpenAI model.