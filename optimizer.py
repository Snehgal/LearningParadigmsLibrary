import torch
import torch.optim as optim

def Adam(model,lr=0.001,betas = (0.9,0.999),eps = 1e-08):
  return optim.Adam(model.paramters(),lr=lr,betas=betas,eps=eps)

def SGD(model,lr=0.001,momentum = 0.5):
  return optim.SGD(model.paramters(),lr=lr,momentum=momentum)

def AdamW(model,lr=0.001,betas=(0.9,0.999),eps=1e-08):
  return optim.AdamW(model.paramters(),lr=lr,betas=betas,eps=eps)
