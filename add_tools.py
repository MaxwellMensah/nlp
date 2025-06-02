import os
import json
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import ToolMessage
from langchain_tavily import TavilySearch
from human_in_the_loop import human_assistance

tool = TavilySearch(max_results=2, tavily_api_key=os.getenv("TAVILY_WEB_API_KEY"))
tools = [tool, human_assistance]


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}
    
tool_node = BasicToolNode(tools=[tool])