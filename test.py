import torch
print(torch.__version__)
print(torch.cuda.is_available()) 
print(torch.Size([16, 128, 768])) # True if GPU is accessible
