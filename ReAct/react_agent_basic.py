from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent,tool
from langchain_community.tools import TavilySearchResults
import datetime

load_dotenv()

model = ChatOpenAI()

search_tool =TavilySearchResults(search_depth="basic")

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S") :
    """Get the current system time in the specified format."""
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

tools = [search_tool,get_system_time]

agent=initialize_agent(tools=tools,llm=model, agent="zero-shot-react-description",verbose=True)#zero-shot means no prior knowledge of the task

agent.invoke("When was SpaceX's recent launch and how many days ago was that from the instant") #invoke the agent with a query