from rich.console import Console
from langchain_openai import ChatOpenAI

console = Console()

# Connect LangChain to your local vLLM container
llm = ChatOpenAI(
    openai_api_base="http://server.domain.com:8000/v1", # Points to your vLLM Docker port
    openai_api_key="EMPTY",                     # vLLM doesn't require a real API key by default
    model_name="nvidia/Qwen3.6-35B-A3B-NVFP4",  # The exact model name vLLM is serving
    max_tokens=2048
)

# Test the connection
query = "Hello, who are you?"
print(f"\nQuestion: {query}")

response = llm.invoke("Hello, who are you?")
console.print(f"Answer: {response.content}", style="white")
print()