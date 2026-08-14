
import os
from rich.console import Console

# For LLM interface
from langchain_openai.llms import OpenAI

# For Chat interface
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai.chat_models import ChatOpenAI


console = Console()
server_fqdn = os.getenv("VLLM_SERVER_FQDN") # vLLM Hostname
if not server_fqdn:
    raise ValueError(
        "Configuration Error: 'VLLM_SERVER_FQDN' environment variable is not set.\n"
        "Please set it in your Ubuntu terminal using:\n"
        "export VLLM_SERVER_FQDN='server.domain.com'"
    )
openai_api_base = f"http://{server_fqdn}:8000/v1"

MODEL_NAME = "nvidia/Qwen3.6-35B-A3B-NVFP4"
MAX_TOKENS = 2048

# LLM Interface: Send a question and get a response / # Qwen3.6 on vLLM
model = OpenAI(openai_api_base=openai_api_base, model=MODEL_NAME, max_tokens=MAX_TOKENS)
question = "The sky is"
response = model.invoke(question)
print()
print(question)
console.print(response, style="white")
print()


# Chat Interface: Send a question and get a response / # Qwen3.6 on vLLM
model = ChatOpenAI(openai_api_base=openai_api_base, model=MODEL_NAME, max_tokens=MAX_TOKENS)


system_msg = SystemMessage('''You are a helpful assistant that responds to questions with three exclamation marks.''')
question = HumanMessage('What is the capital of France?')
response = model.invoke([system_msg, question])
print()
print(question)
console.print(response, style="white")
print()




"""
# Connect LangChain to your Qwen3.6 on vLLM
model = ChatOpenAI(openai_api_base=openai_api_base, openai_api_key="EMPTY", model_name=MODEL_NAME, max_tokens=MAX_TOKENS)

# Test the connection
query = "Hello, who are you?"
print(f"\n1. Question: {query}")
response = model.invoke(query)
console.print(f"Answer: {response.content}", style="white")

# HumanMessage
query = '[HumanMessage("What is the capital of France?")]'
print(f"\n2. Question: {query}")
response = model.invoke(query)
console.print(f"Answer: {response.content}", style="white")

# SystemMessage / HumanMessage
system_msg = SystemMessage('''You are a helpful assistant that responds to questions with three exclamation marks.''')
human_msg = HumanMessage('What is the capital of France?')
print(f"\n3. System: {system_msg}")
print(f"Question: {human_msg}")
response = model.invoke([system_msg, human_msg])
console.print(f"Answer: {response.content}", style="white")

print()
"""





