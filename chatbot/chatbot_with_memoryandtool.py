from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode


load_dotenv()

memory = MemorySaver()

llm = ChatOpenAI()

class BasicChatState(TypedDict): 
    messages: Annotated[list, add_messages]

search_tool=TavilySearch(max_results=2)
tools=[search_tool]
llm_with_tools=llm.bind_tools(tools=tools)

def chatbot(state: BasicChatState): 
    return {
       "messages": [llm_with_tools.invoke(state["messages"])]
    }

def tools_router(state: BasicChatState):
    last_message=state["messages"][-1]

    if(hasattr(last_message,"tool_calls") and len(last_message.tool_calls)>0):#does the last message has tool call property in it and is length of toll call is >0
        return "tool_node"
    else: 
        return END
    
tool_node=ToolNode(tools=tools)

graph = StateGraph(BasicChatState)

graph.add_node("chatbot", chatbot)
graph.add_node("tool_node",tool_node)

graph.add_conditional_edges("chatbot",tools_router)
graph.add_edge("tool_node","chatbot")

graph.set_entry_point("chatbot")

app = graph.compile(checkpointer=memory)

config = {"configurable": {
    "thread_id": 1
}}

while True: 
    user_input = input("User: ")
    if(user_input in ["exit", "end"]):
        break
    else: 
        result = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        }, config=config)

        print("AI: " + result["messages"][-1].content)