# flake8: noqa
from langchain.output_parsers.regex import RegexParser
from langchain_core.prompts import HumanMessagePromptTemplate, PromptTemplate

output_parser = RegexParser(
    regex=r"(.*?)\nScore: (\d*)",
    output_keys=["answer", "score"],
)

prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, you can look it up on the internet but you must output it like this: 
"Oh our database doesn't have that answer. However I found the following answer to your question [answer] [source_of_answer]

{context}
Question: {question}
"""
HUMAN_MESSAGE_CUSTOM_PROMPT = HumanMessagePromptTemplate.from_template(
    template=prompt_template,
    input_variables=["context", "question"],
    output_parser=output_parser,
)

CUSTOM_PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"],
    output_parser=output_parser,
)
