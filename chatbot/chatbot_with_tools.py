from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated
from langchain_core.messages import AIMessage,HumanMessage
from langgraph.graph import add_messages,StateGraph,END
from sqlalchemy import true
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

load_dotenv()

llm = ChatOpenAI()

class BasicChatState(TypedDict):
    messages: Annotated[list,add_messages]

search_tool=TavilySearch(max_results=2)
tools=[search_tool]

llm_with_tools=llm.bind_tools(tools=tools)

def chatbot(state: BasicChatState):
    return {
        "messages": [llm_with_tools.invoke(state['messages'])]
    }

def tools_router(state: BasicChatState):
    last_message=state["messages"][-1]

    if(hasattr(last_message,"tool_calls") and len(last_message.tool_calls)>0):#does the last message has tool call property in it and is length of toll call is >0
        return "tool_node"
    else: 
        return END
    
tool_node=ToolNode(tools=tools) # if using messagesss instead of messages add another patra meter messages_key

graph=StateGraph(BasicChatState)

graph.add_node("chatbot",chatbot)
graph.add_node("tool_node",tool_node)
graph.set_entry_point("chatbot")

graph.add_conditional_edges("chatbot",tools_router)
graph.add_edge("tool_node","chatbot") 

app=graph.compile()


while True:
    user_input=input("User: ")
    if(user_input in ["exit","end"]):
        break
    else:
        result=app.invoke({
            "messages":[HumanMessage(content=user_input)]
        })
    
    print(result)