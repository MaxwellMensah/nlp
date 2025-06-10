import random
from typing import Literal, Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from memory import memory
from add_tools import tools
from langgraph.prebuilt import ToolNode, tools_condition
from langsmith import traceable
from models import available_models


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
llm_with_tools = llm.bind_tools(tools)

# routes, commented this out because i'm using the pre-built tool
# def route_tools(
#     state: State,
# ):
#     """
#     Use in the conditional_edge to route to the ToolNode if the last message
#     has tool calls. Otherwise, route to the end.
#     """
#     if isinstance(state, list):
#         ai_message = state[-1]
#     elif messages := state.get("messages", []):
#         ai_message = messages[-1]
#     else:
#         raise ValueError(f"No messages found in input state to tool_edge: {state}")
#     if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
#         return "use_route_tools"
#     return "end"

# Initialize the StateGraph
@traceable(run_name="OpenAI Chatbot Invocation", run_type="llm")
def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    assert(len(message.tool_calls) <= 1)
    return {"messages": [llm_with_tools.invoke(state["messages"])]}



def state_graph(state: State):
    # Build graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_edge(START, "chatbot")
    
    # Adding conditional edges to the graph
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
        # {
        #     "use_route_tools": "my_tools", 
        #     "end": END
        # }
    )

    # Any time a tool is called, we return to the chatbot to decide the next step
    # Add edge from tools back to chatbot
    graph_builder.add_edge("tools", "chatbot")
    
    # Compile graph
    return graph_builder.compile(
        checkpointer=memory
    )

# Export the graph variable so it can be imported elsewhere
graph = state_graph(State()) 