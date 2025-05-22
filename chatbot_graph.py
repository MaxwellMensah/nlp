import random
from typing import Literal, Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
# from langchain_openai import ChatOpenAI  # <-- Import OpenAI LLM

from langsmith import traceable
from models import available_models 
# from langchain_tavily import TavilySearch


class State(TypedDict):
    messages: Annotated[list, add_messages]


# Function to get a working LLM model, fallback if one fails
def get_llm():
    models = available_models()
    try:
        # Try OpenAI model first
        _ = models.openai_model.invoke(["ping"])
        return models.openai_model
    except Exception:
        try:
            # Fallback to Gemini model
            _ = models.gemini_model.invoke(["ping"])
            return models.gemini_model
        except Exception:
            raise RuntimeError("No available LLM models.")

llm = get_llm()

# tool = TavilySearch(max_results=2)
# tools = [tool]
# # tool.invoke("What's a 'node' in LangGraph?")

llm_with_tools = llm.bind_tools(tools)

# Initialize the StateGraph
@traceable(run_name="OpenAI Chatbot Invocation", run_type="llm")
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

def state_graph(state: State):
    # Build graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    # Compile graph
    return graph_builder.compile()

# Export the graph variable so it can be imported elsewhere
graph = state_graph(State())
