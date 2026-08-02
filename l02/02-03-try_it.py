import os
from rich.console import Console
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

def create_wikipedia_tool():
    """
    Create a Wikipedia search tool for querying encyclopedic information.
    
    The tool should be configured with:
    - top_k_results: 3 (return top 3 Wikipedia articles)
    - doc_content_chars_max: 2000 (limit content length per result)
    
    Returns:
        WikipediaQueryRun: A configured Wikipedia search tool instance
    
    Hint: WikipediaQueryRun takes an api_wrapper parameter 
    """
    # Create the Wikipedia API wrapper
    api_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=2000)

    # Create and return the Wikipedia search tool
    return WikipediaQueryRun(api_wrapper=api_wrapper)


# LLM Variables
server_fqdn = os.getenv("VLLM_SERVER_FQDN")
if not server_fqdn:
    raise ValueError("Configuration Error: 'VLLM_SERVER_FQDN' environment variable is not set.\n")
openai_api_base = f"http://{server_fqdn}:8000/v1"

MODEL_NAME = "nvidia/Qwen3.6-35B-A3B-NVFP4"
TEMPERATURE = 0.1
MAX_TOKENS = 2048

console = Console()

# Load API keys fron environment variables
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Load model (Qwen3.6 on vLLM)
model = ChatOpenAI(openai_api_base=openai_api_base, openai_api_key="EMPTY", model_name=MODEL_NAME, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)

# Instantiate Tavily search tool
#search_tool = TavilySearch(max_results=5, search_depth="basic", include_raw_content=False, include_images=False)

# Create the agent with the search tool
#agent = create_agent(model=model, tools=[search_tool])

console.print("\nAPI keys loaded", style="gold1")
console.print(f"Model configured: {model.model_name}\n", style="gold1", highlight=False)
#console.print(f"Tavily search configured: {search_tool.name}", style="gold1")
#console.print("Agent created\n", style="gold1")


# Test the AI Agent
#result1 = generate_and_print_response(agent, "What are the latest developments in artificial intelligence in 2026?")



