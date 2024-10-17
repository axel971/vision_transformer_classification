import numpy as np
import torch.nn as nn
import torch
import torchmetrics
from tqdm import tqdm
from model.testing_functions import eval

def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               metric_fn: torchmetrics,
               device: torch.device):

    loss_value = 0
    
    progressBar_dataloader = tqdm(dataloader)

    model.train()
    
    for (X, y) in progressBar_dataloader:
        
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        loss_value += loss.item()
        
        #print(y_pred.shape)
        #print(y.shape)

        metric = metric_fn(y_pred, y)
        
        progressBar_dataloader.set_postfix(loss = loss.item(), metric =  metric.item())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    loss_value /= len(dataloader)
    metric_value = metric_fn.compute()

    metric_fn.reset()

    return loss_value, metric_value

def train(model: nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: nn.Module,
          metric_fn: torchmetrics,
          epochs: int,
          device: torch.device):

    results = {"Train loss": [],
               "Train metric": [],
               "Test loss": [],
               "Test metric": []}


    for epoch in range(epochs):

        print(f"Epoch {epoch + 1}/{epochs}: ")

        train_loss_value, train_metric_value = train_step(model = model,
                                              dataloader = train_dataloader,
                                              optimizer = optimizer,
                                              loss_fn = loss_fn,
                                              metric_fn = metric_fn,
                                              device = device)

        test_loss_value, test_metric_value = eval(model = model,
                                                 dataloader = test_dataloader,
                                                  loss_fn = loss_fn,
                                                  metric_fn = metric_fn,
                                                  device = device)

        results["Train loss"].append(train_loss_value)
        results["Train metric"].append(train_metric_value)
        results["Test loss"].append(test_loss_value)
        results["Test metric"].append(test_metric_value)

        print(f"Epoch {epoch + 1} | Train loss: {train_loss_value} | Train metric: {train_metric_value} | Test loss: {test_loss_value} | Test metric: {test_metric_value} ")

    return results
