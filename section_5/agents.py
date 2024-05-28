from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationSummaryMemory
from langchain_core.messages import SystemMessage
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate

from section_5.prompts import custom_system_prompt
from section_5.tools import chat, search_question_in_vector_store_tool, search_from_google_tool, get_user_context_tool

memory = ConversationSummaryMemory(
    memory_key="messages",
    return_messages=True,
    llm=chat
)

chat_prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(content=custom_system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template(
            "{content}"
        ),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ],
    input_variables=["content", "messages", "agent_scratchpad", "user_context"],
)

tools = [search_question_in_vector_store_tool, search_from_google_tool, get_user_context_tool]

agent = create_openai_functions_agent(
    llm=chat,
    prompt=chat_prompt,
    tools=tools,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory
)

while True:
    message = input("Ask me a question: ")
    print(
        agent_executor.invoke(
            {
                "content": message,
            },
        )["output"]
    )
