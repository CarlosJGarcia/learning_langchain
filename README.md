**Instalación del entorno** \
conda create -n langchain python=3.11 -y \
conda activate langchain \
conda install -c conda-forge faiss-gpu -y \
pip install -U langchain langchain-core langchain-community langgraph langsmith \ 
pip install -U langchain-openai python-dotenv jupyterlab pydantic tiktoken
