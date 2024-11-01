# utils.py

import torch

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path, device):
    model = initialize_model().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model
