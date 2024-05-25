import os
from dotenv import load_dotenv
from langchain_openai import OpenAI

load_dotenv()
# Create a new instance of OpenAI
llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
result = llm("What is the capital of France?")
print(result)
