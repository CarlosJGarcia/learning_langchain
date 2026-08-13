import os
import ssl
import urllib3
import requests
from rich.console import Console

# HTTPS -> HTTP for Tavily and LangChain to prevent Corporate HTTP Proxy issues
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ""
os.environ['REQUESTS_CA_BUNDLE'] = ""
original_request = requests.Session.request
def unverified_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)
requests.Session.request = unverified_request


from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent

#Helper function for search
def generate_and_print_response(agent, query):
    """
    Invoke the agent with a query and print the response.
    
    Args:
        agent: The LangChain agent instance
        query: The user query string
    """

    # Print the query
    print(query)
    
    # Invoke the agent with the query
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})

    # DEBUG START
    """
    console.print("\n--- Agent Internal Monologue ---", style="magenta")
    for msg in result["messages"]:
        msg_type = msg.__class__.__name__
        
        # If the model made a tool call, print it
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            console.print(f"[{msg_type}] Tool Call Requested: {msg.tool_calls}", style="yellow")
            
        # Print the content of the message (truncate tool results slightly for readability)
        content = str(msg.content)
        if len(content) > 500 and msg_type == "ToolMessage":
            content = content[:500] + "... [TRUNCATED]"
            
        console.print(f"[{msg_type}]: {content}\n", style="white")
    """
    # DEBUG END
    
    # Extract and print the final response
    final_message = result["messages"][-1]
    
    console.print(f"Response: {final_message.content}", style="white", highlight=False)
    
    return result


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
if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY not found in environment variables")


# Load model (Qwen3.6 on vLLM)
model = ChatOpenAI(openai_api_base=openai_api_base, openai_api_key="EMPTY", model_name=MODEL_NAME, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)

# Instantiate Tavily search tool
search_tool = TavilySearch(max_results=5, search_depth="basic", include_raw_content=False, include_images=False)

# Create the agent with the search tool
agent = create_agent(model=model, tools=[search_tool])

console.print("\nAPI keys loaded", style="gold1")
console.print(f"Model configured: {model.model_name}", style="gold1", highlight=False)
console.print(f"Tavily search configured: {search_tool.name}", style="gold1")
console.print("Agent created\n", style="gold1")


# Test Tavily
"""
console.print("\n--- Testing Tavily Directly ---", style="cyan")
try:
#    # We pass a dictionary with the expected 'query' argument
    test_result = search_tool.invoke({"query": "Latest AI developments 2026"})
    console.print(test_result)
except Exception as e:
    console.print(f"Tavily crashed! Error: {e}", style="red")
"""

# Run the AI Agent
result1 = generate_and_print_response(agent, "What are the latest developments in artificial intelligence in 2026?")



