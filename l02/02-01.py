# Competency 2.1 Running Your First Pre-Built Agent

import os
from rich.console import Console
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool


# Define a tool for the pre-built agent
# Mock implementation - in production, this function would call a real weather API
@tool
def get_weather(location: str) -> str:
    """Get current weather for a location.
    
    Args:
        location: The city or location to get weather for
        
    Returns:
        A string describing the current weather conditions, or a message if data is not available
    """    
    weather_data = {
        "new york": "Sunny, 22°C",
        "london": "Cloudy, 19°C",
        "tokyo": "Clear, 18°C",
    }

    location_key = location.lower().strip()

    if location_key in weather_data:
        return f"Weather in {location}: {weather_data[location_key]}"
    else:
        return f"Weather data not available for {location}"


#Helper function to ask the agent a question and print the answer.
def ask_agent(question: str):
    
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    # Extract and print the final answer
    final_message = result["messages"][-1]
    console.print(final_message.content, style="white", highlight=False)


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
TEMPERATURE = 0.1
MAX_TOKENS = 2048


# Connect LangChain to your Qwen3.6 on vLLM
model = ChatOpenAI(openai_api_base=openai_api_base, openai_api_key="EMPTY", model_name=MODEL_NAME, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)

# Initialize agent and execute agent flow (ReAct Loop)
agent = create_agent(model, tools=[get_weather])
query = "weather in tokio today?"
print(query)
ask_agent(query)
print()

query = "weather in pamplona today?"
print(query)
ask_agent(query)
print()



