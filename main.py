import os
from dotenv import load_dotenv
load_dotenv()
from langsmith import traceable
from typing import Annotated, TypedDict
from chatbot_graph import graph

from langsmith import utils
utils.get_env_var.cache_clear()


def display_graph(graph):
    """Display the graph image if possible."""
    try:
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception as e:
        print(f"Could not display graph: {e}")


@traceable(run_name="Stream Graph Updates", metadata={"llm": "gemini-2.0-flash-lite-001"})
def stream_graph_updates(graph, user_input: str):
    """Stream updates from the graph based on user input."""
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
            print("-" * 100)
            print("\n")


def main():
    """Main function to handle user interaction."""
    # Example graph initialization (uncomment and customize as needed)
    # graph = StateGraph(State, start=START, end=END)
    # display_graph(graph)

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye! Talk later")
                break
            # Uncomment the following line after initializing `graph`
            stream_graph_updates(graph, user_input)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            # Fallback input for environments without input()
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            # Uncomment the following line after initializing `graph`
            stream_graph_updates(graph, user_input)
            break

if __name__ == "__main__":
    main()
