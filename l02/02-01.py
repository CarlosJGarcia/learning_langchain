# Competency 2.1 Running Your First Pre-Built Agent
# AI Agent developed using LangChain
# AI Agent architecture pattern: ReAct Loop 

import os
from rich.console import Console
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool


# Define a tool for the Agent
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
    
    # agent.invoke hace dos cosas:
    # - Construye un fichero JSON en formato OpenAI describiendo cada una de las herramientas disponibles (en teste caso, get_weather)
    # - Hace una llamada al LLM pasando como parámetro la query y la lista de herramientas
    # - El LLM, que ha sido fine-tuned para entender este JSON de OpenAI, responde con otro JSON para que se ejecute la herramienta
    # - agent.invoke ejecuta el código Python de la herramienta, recoge el resultado y se lo devuelve al LLM en otro JSON de formato OpenAI
    # - El LLM parsea, ese nuevo JSON y junto con la información de contexto, calcula la frase de respuesta, que envia de vuelta
    # - agent.invoke devuelve el resultado al agente AI (este código)
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    # Extract and print the final answer
    final_message = result["messages"][-1]
    clean_message = final_message.content.strip()                 # Remove trailing \n in the LLM response
    console.print(clean_message, style="white", highlight=False)


console = Console()

# Define model settings (Qwen3.6 on vLLM)
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


# Load model (Qwen3.6 on vLLM)
model = ChatOpenAI(openai_api_base=openai_api_base, openai_api_key="EMPTY", model_name=MODEL_NAME, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)

# Initialize agent 
agent = create_agent(model, tools=[get_weather])

# create_agent() from langchain.agents Internally implements a ReAct loop.
# When agent.invoke() is called from ask_agent() the agent runs an internal loop of:
#   - Think: the LLM reasons about what to do
#   - Act: calls a tool (e.g., get_weather)
#   - Observe: receives the tool's result
#   - Loops back to Think until it produces a final answer
# There is no need for a for/while loop in the Python code, the looping behavior is encapsulated inside the agent.
# A ReAct Chain (linear) would require to manually implement the think→act→observe steps in the Python code.
# The ReAct Loop delegates the iteration to the agent framework


# Agent flow (ReAct Loop) positive case
query = "What's the weather like in tokio today?"
print(f"\n{query}")
ask_agent(query)
print()

# Agent flow (ReAct Loop) negative case
query = "What's the weather like in pamplona today?"
print(query)
ask_agent(query)
print()



