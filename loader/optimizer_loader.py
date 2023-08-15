import torch

def load_optimizer(optimizer_name,model,lrate):
    optimizer = None
    if optimizer_name == "SGD":
       optimizer = torch.optim.SGD(model.parameters(),lr = lrate,momentum = 0.9)
    
    return optimizer
    