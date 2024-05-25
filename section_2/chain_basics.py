from langchain.chains import LLMChain, SequentialChain
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
    prompt=code_prompt,
    output_key="code"
)

task = input("Enter a code task you want to generate code for: ")
language = input("Enter the language you want to generate code for: ")

code_prompt_2 = PromptTemplate(
    template="Write a the code to test for the following {language} code:\n{code}",
    input_variables=["language", "code"],
)

code_chain_part_2 = LLMChain(
    llm=llm,
    prompt=code_prompt_2,
    output_key="test"
)

chain = SequentialChain(
    chains=[code_chain_part_1, code_chain_part_2],
    input_variables=["language", "task"],
    output_variables=["code", "test"]
)

result = chain({
    "language": language,
    "task": task
})

print("Generated code and test:")
print(result["code"])
print(result["test"])

