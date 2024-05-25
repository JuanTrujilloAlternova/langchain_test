from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

from settings import OPENAI_KEY

llm = OpenAI(openai_api_key=OPENAI_KEY)
code_prompt = PromptTemplate(
    template="Write a very short {language} program function that will {task}",
    input_variables=["language", "task"],
)

code_chain_part_1 = LLMChain(
    llm=llm,
    prompt=code_prompt
)

task = input("Enter a code task you want to generate code for: ")
language = input("Enter the language you want to generate code for: ")

result = code_chain_part_1(
    {
        "language": language,
        "task": task
    }
)

print(result)
