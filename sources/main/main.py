# python main.py --data_dir "C:\Users\axell\Documents\dev\vision_transformer_classification\data"

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision import datasets
from torchvision.transforms import ToTensor
import argparse
from pathlib import Path
from torchmetrics import Accuracy

import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from model.training_functions import train
from model.ViT import ViT

def main(data_dir: str):
    
    DATA_DIR = Path(data_dir)

    # Import Dataset
    training_dataset = datasets.CIFAR100(root = DATA_DIR,
                                     train = True,
                                     download = True,
                                     transform = ToTensor())

    testing_dataset = datasets.CIFAR100(root = DATA_DIR,
                                    train = False,
                                    download = True,
                                    transform = ToTensor())
    class_names = training_dataset.classes
    num_classes = len(class_names)
    
    img, _ = training_dataset[0]
    img_size = img.size(-1)

    print(num_classes)
    
    # Create DataLoader
    BATCH_SIZE = 248
    train_dataloader = DataLoader(dataset = training_dataset,
                                     batch_size = BATCH_SIZE,
                                     shuffle = True)
    test_dataloader = DataLoader(dataset = testing_dataset,
                                    batch_size = BATCH_SIZE,
                                    shuffle = False)

    # Instanciate device agnostic code
    device = "gpu" if torch.cuda.is_available() else "cpu"

    # Instantiate the model
    vit = ViT(img_size = img_size,
              patch_size = 8,
              num_classes = num_classes).to(device)    

    # Instantiate the optimizer, loss_function, and metric
    optimizer = Adam(params = vit.parameters(),
                     lr = 1e-3)
    loss_fn = CrossEntropyLoss()
    metric_fn = Accuracy(task = "multiclass", num_classes = num_classes)

    # Train the model
    epochs = 60

    results = train(model = vit,
                    train_dataloader = train_dataloader,
                    test_dataloader = test_dataloader,
                    optimizer = optimizer,
                    loss_fn = loss_fn,
                    metric_fn = metric_fn,
                    epochs = epochs,
                    device = device)
    return

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required = True, help = "Path toward data directory")
    args = parser.parse_args()
    main(data_dir = args.data_dir)
