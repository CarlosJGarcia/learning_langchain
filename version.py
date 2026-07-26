import sys
import faiss
import langchain

print()
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print(f"LangChain version: {langchain.__version__}")

# Check CUDA-capable GPUs for FAISS 
num_gpus = faiss.get_num_gpus()
if num_gpus > 0:
    res = faiss.StandardGpuResources()  # Initialize GPU resources
    print("FAISS GPU available")
else:
    print("CUDA not available")

print()