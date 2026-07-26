import sys
import faiss
import langchain
# import langgraph
# import langsmith
# from langchain_openai import ChatOpenAI



print()
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print(f"LangChain version: {langchain.__version__}")
# print(f"LangGraph version: {langgraph.__version__}")



# Check how many CUDA-capable GPUs FAISS detects
num_gpus = faiss.get_num_gpus()
print(f"GPUs available for FAISS: {num_gpus}")

# Quick sanity check
if num_gpus > 0:
    res = faiss.StandardGpuResources()  # Initialize GPU resources
    print("FAISS GPU successfully initialized on your RTX 3090!")
else:
    print("CUDA support not detected.")





# print("PyTorch version:", torch.__version__)
# print("Apple Silicon acceleration:", torch.backends.mps.is_available())
# print("CUDA enabled:", torch.cuda.is_available())
"""
if torch.cuda.is_available():
    print("CUDA Compute Platform.", torch.version.cuda)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print(f"CUDA not available, using CPU.") 
"""
print()