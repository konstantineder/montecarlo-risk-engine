import torch

# if torch.backends.mps.is_available():
#     device = torch.device("mps")
#     FLOAT = torch.float32 # MPS backend works only with float32
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     FLOAT = torch.float64 # Use float64 for better precision on GPU
#else:
device = torch.device("cpu")
FLOAT = torch.float64 # Use float64 for better precision on CPU
