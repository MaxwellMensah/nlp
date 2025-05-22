import google.generativeai as genai
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI  #pip install langchain_google_genai

class available_models:
    def __init__(self):
        self.openai_model = self.openai_model()
        self.gemini_model = self.gemini_model()
        
    def gemini_model(self):
        """Initialize Gemini LLM (adjust model name as needed)."""
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite-001",  # or "gemini-2" if you have access
            temperature=0,
            max_tokens=None,
            max_retries=2,
        )
        return llm
    
    def openai_model(self):
        """Initialize OpenAI LLM (adjust model name as needed)."""
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",  # or "gpt-4" if you have access
            temperature=0,
            max_tokens=None,
            max_retries=2,
        )
        return llm
