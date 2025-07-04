from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage
from langgraph.graph import add_messages, StateGraph, END
from langchain_openai import ChatOpenAI

class State(TypedDict): 
    messages: Annotated[list, add_messages]

llm = ChatOpenAI()

GENERATE_POST = "generate_post"
GET_REVIEW_DECISION = "get_review_decision"
POST = "post"
COLLECT_FEEDBACK = "collect_feedback"

def generate_post(state: State): 
    return {
        "messages": [llm.invoke(state["messages"])]
    }

def get_review_decision(state: State):  
    post_content = state["messages"][-1].content 
    
    print("\nCurrent LinkedIn Post:\n")
    print(post_content)
    print("\n")

    decision = input("Post to LinkedIn? (yes/no): ")

    if decision.lower() == "yes":
        return POST
    else:
        return COLLECT_FEEDBACK


def post(state: State):  
    final_post = state["messages"][-1].content  
    print("\nFinal LinkedIn Post:\n")
    print(final_post)
    print("\n Post has been approved and is now live on LinkedIn!")

def collect_feedback(state: State):  
    feedback = input("How can I improve this post?")
    return {
        "messages": [HumanMessage(content=feedback)]
    }

graph = StateGraph(State)

graph.add_node("generate_post", generate_post)
graph.add_node("get_review_decision", get_review_decision)
graph.add_node("collect_feedback", collect_feedback)
graph.add_node("post", post)

graph.set_entry_point("generate_post")

graph.add_conditional_edges("generate_post", get_review_decision)
graph.add_edge("post", END)
graph.add_edge("collect_feedback", "generate_post")

app = graph.compile()

response = app.invoke({
    "messages": [HumanMessage(content="Write me a LinkedIn post on AI Agents taking over content creation")]
})






 




