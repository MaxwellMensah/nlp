import os
from dotenv import load_dotenv
from langsmith import traceable
from chatbot_graph import graph
from langgraph.types import Command
from memory import memory
from IPython.display import Image, display

from langsmith import utils
utils.get_env_var.cache_clear()
load_dotenv()

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except ImportError:
    print("Missing dependencies for graph visualization. Please install the required packages.")
except Exception as e:
    print(f"An error occurred while displaying the graph: {e}")


config={
    "configurable": 
        {
            "thread_id": "1", 
            "checkpoint_ns": "my_checkpoint_ns", 
            "checkpoint_id": "my_checkpoint_id", 
            "checkpointer": memory
         }
    }

@traceable(run_name="Stream Graph Updates", metadata={"llm": "gemini-2.0-flash-lite-001"})
def stream_graph_updates(user_input: str):
    """Stream updates from the graph based on user input."""
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}, config=config):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
      
      
@traceable(run_name="Human Graph Updates", metadata={"llm": "gemini-2.0-flash-lite-001"})      
def human_feedback(graph, user_input: str):  
    """Stream updates from the graph based on human feedback."""      
    human_query = user_input[7:].strip()
    events = graph.stream(
        {"messages": [{"role": "user", "content": human_query}]},
        config=config,
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()
    human_response = (
    "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
    " It's much more reliable and extensible than simple autonomous agents."
    )
    print("Assistant:", human_response)
    human_command = Command(resume={"data": human_response})
    events = graph.stream(human_command, config=config, stream_mode="values")
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()            
            

def main():
    """Main function to handle user interaction with Tavily search and human in the loop."""

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye! Talk later")
                break
            elif user_input.startswith("!human"):
                human_feedback(graph, user_input)
            else:
                stream_graph_updates(user_input=user_input)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            # Fallback input for environments without input()
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(graph, user_input)
            break
        
if __name__ == "__main__":
    main()