import os
from rich.console import Console
from langchain_openai import ChatOpenAI


console = Console()

server_fqdn = os.getenv("VLLM_SERVER_FQDN")
if not server_fqdn:
    raise ValueError(
        "Configuration Error: 'VLLM_SERVER_FQDN' environment variable is not set.\n"
        "Please set it in your Ubuntu terminal using:\n"
        "export VLLM_SERVER_FQDN='server.domain.com'"
    )
openai_api_base = f"http://{server_fqdn}:8000/v1"

MODEL_NAME = "nvidia/Qwen3.6-35B-A3B-NVFP4"
MAX_TOKENS = 2048

# Connect LangChain to your Qwen3.6 on vLLM
llm = ChatOpenAI(openai_api_base=openai_api_base, openai_api_key="EMPTY", model_name=MODEL_NAME, max_tokens=MAX_TOKENS)

# Test the connection
query = "Hello, who are you?"
print(f"\nQuestion: {query}")

response = llm.invoke("Hello, who are you?")
console.print(f"Answer: {response.content}", style="white")
print()