from xml.dom import XMLNS_NAMESPACE
from langchain_openai import ChatOpenAI
from langchain.agents import tool,create_react_agent
import datetime
from langchain_community.tools import TavilySearchResults
from langchain_tavily import TavilySearch
from langchain import hub

llm = ChatOpenAI()

search_tool =TavilySearch(search_depth="basic")

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S") :
    """Get the current system time in the specified format."""
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

tools = [search_tool,get_system_time]

react_prompt=hub.pull("hwchase17/react")

react_agent_runnable=create_react_agent(tools=tools,llm=llm,prompt=react_prompt) 
