**Instalación del entorno** \
conda create -n langchain python=3.11 -y \
conda activate langchain \
conda install -c pytorch -c nvidia faiss-gpu -y \
pip install -U langchain langchain-core langchain-community langgraph langsmith \
pip install -U langchain-openai python-dotenv jupyterlab pydantic tiktoken \
conda install rich -y -c conda-forge \
pip install langchain-tavily \
pip install wikipedia \
pip install -U langchain-postgres
