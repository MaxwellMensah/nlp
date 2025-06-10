import os
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
load_dotenv()

tool = TavilySearch(
    max_results=2, tavily_api_key=os.getenv("TAVILY_WEB_API_KEY")
    )