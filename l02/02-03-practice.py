import os
from rich.console import Console
from langchain_openai import ChatOpenAI

# from dotenv import load_dotenv
# from langchain.agents import create_agent
# from langchain_tavily import TavilySearch


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
if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY not found in environment variables")


# Load model (Qwen3.6 on vLLM)
model = ChatOpenAI(openai_api_base=openai_api_base, openai_api_key="EMPTY", model_name=MODEL_NAME, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)


print("\nAPI keys loaded")
print(f"Model configured: {model.model_name}\n")


