from langchain.chat_models import init_chat_model
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

model = init_chat_model("gpt-4o-mini", model_provider="openai")
#model = init_chat_model("claude-3-7-sonnet-latest", model_provider="anthropic")
#model = init_chat_model("llama3.1:latest", model_provider="ollama")

from langchain_core.prompts import PromptTemplate
from langchain.agents import (
    create_react_agent,
    AgentExecutor
)

from langchain import hub

template = """"
    Return all articles from eduklein.com.br saved in my Readwise account.
    """

prompt_template = PromptTemplate(
    template=template
)

from clients.readwise import search_readwise_articles
tools_for_agent = [ search_readwise_articles ]

react_prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm=model, tools=tools_for_agent, prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent)

result = agent_executor.invoke(
    input={"input": prompt_template}
)

print(result["output"])