import os
# from dotenv import load_dotenv
# from langchain.agents import create_agent
# from langchain_openai import ChatOpenAI
# from langchain_tavily import TavilySearch

# Load environment variables from .env file
# load_dotenv()

# Verify API keys are loaded
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")
if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY not found in environment variables")

print("API keys loaded successfully!")