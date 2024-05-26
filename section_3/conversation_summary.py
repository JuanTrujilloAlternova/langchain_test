from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory.summary import ConversationSummaryMemory
from langchain_core.prompts import (
    HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate,
    MessagesPlaceholder
)
from settings import OPENAI_KEY

chat = ChatOpenAI(openai_api_key=OPENAI_KEY)

# With just buffer memory will be deleted after the program ends
# Summary needs llm to parse the messages
memory = ConversationSummaryMemory(
    memory_key="messages",
    return_messages=True,
    llm=chat
)


chat_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a bot named Nancy, you are super shy but friendly. Every time you answer"
            "a question relate that question to Naruto, your favorite anime."
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
