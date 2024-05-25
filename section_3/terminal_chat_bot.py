from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import (
    HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate
)
from settings import OPENAI_KEY

chat = ChatOpenAI(openai_api_key=OPENAI_KEY)
chat_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a bot who only understands spanish, if you are asked in any other language"
            "you shouldn't understand."
        ),
        HumanMessagePromptTemplate.from_template(
            "{content}"
        )
    ],
    input_variables=["content"],
)

chain = LLMChain(
    llm=chat,
    prompt=chat_prompt,
)

while True:
    message = input("Ask me a question: ")
    # TODO: add a way to store the conversation history
    result = chain({
        "content": message
    })
    print(result["text"])
