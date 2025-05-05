from langchain.chat_models import init_chat_model

from dotenv import load_dotenv

# load environment variables from .env file, mostly API keys
load_dotenv()

model = init_chat_model("deepseek-chat", model_provider="deepseek")
#model = init_chat_model("claude-3-7-sonnet-latest", model_provider="anthropic")
#model = init_chat_model("gpt-4o-mini", model_provider="openai")

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

from clients.readwise import search_readwise_articles
tools = [ search_readwise_articles ]

agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

query = """"
    Return all articles from eduklein.com.br saved in my Readwise account that explores the Ivy Lee Method.
    """

result = agent_executor.invoke({"input": query})

print(result["output"])