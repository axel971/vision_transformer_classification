import torch
from torch import nn as nn
import torchmetrics

def eval(model: nn.Module,
         dataloader: torch.utils.data.DataLoader,
         loss_fn: nn.Module,
         metric_fn: torchmetrics,
         device: torch.device
         ):

    loss_value = 0

    model.eval()
    
    with torch.inference_mode():
        for (X, y) in dataloader:
            
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            loss = loss_fn(y_pred, y)
            loss_value += loss.item()

            metric_fn(y_pred, y)

    loss_value /= len(dataloader)
    metric_value = metric_fn.compute()

    metric_fn.reset()

    return loss_value, metric_value


