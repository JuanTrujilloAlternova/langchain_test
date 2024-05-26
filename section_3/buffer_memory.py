from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import (
    HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate,
    MessagesPlaceholder
)
from settings import OPENAI_KEY

chat = ChatOpenAI(openai_api_key=OPENAI_KEY)

# Memory key must be the same in ChatPromptTemplate and ConversationBufferMemory
# Because it accesses the memory using the key
memory = ConversationBufferMemory(memory_key="messages", return_messages=True)


chat_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a bot who only understands spanish, if you are asked in any other language"
            "you shouldn't understand."
        ),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template(
            "{content}"
        )
    ],
    input_variables=["content", "messages"],
)

chain = LLMChain(
    llm=chat,
    prompt=chat_prompt,
    memory=memory
)

while True:
    message = input("Ask me a question: ")
    result = chain({
        "content": message
    })
    print(result["text"])
