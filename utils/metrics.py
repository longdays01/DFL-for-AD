import torch

def RMSE(y_preds, y_trues):
    with torch.no_grad():
        return torch.sqrt(torch.mean((y_preds - y_trues)**2))

def MAE(y_preds, y_trues):
    with torch.no_grad():
        return torch.mean(torch.abs(y_preds - y_trues))
