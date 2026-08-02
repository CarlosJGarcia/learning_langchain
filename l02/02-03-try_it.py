import os
from rich.console import Console
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent


# LLM Variables
server_fqdn = os.getenv("VLLM_SERVER_FQDN")
if not server_fqdn:
    raise ValueError(
        "Configuration Error: 'VLLM_SERVER_FQDN' environment variable is not set.\n"
        "Please set it in your Ubuntu terminal using:\n"
        "export VLLM_SERVER_FQDN='server.domain.com'"
    )
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
console.print(f"Model configured: {model.model_name}", style="gold1", highlight=False)
#console.print(f"Tavily search configured: {search_tool.name}", style="gold1")
#console.print("Agent created\n", style="gold1")


# Test the AI Agent
#result1 = generate_and_print_response(agent, "What are the latest developments in artificial intelligence in 2026?")



