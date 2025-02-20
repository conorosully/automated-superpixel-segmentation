# Training unet for coastline detection
# Conor O'Sullivan
# 07 Feb 2023

# Imports
import numpy as np
import pandas as pd
import random
import glob
import argparse
import os

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from network import U_Net, R2U_Net, AttU_Net, R2AttU_Net

import utils


def main():
    # Define the argument parser
    parser = argparse.ArgumentParser(description="Train a model on specified dataset")

    # Adding arguments
    parser.add_argument("--model_name", type=str, help="Name of the model to train")
    parser.add_argument("--sample", action="store_true", help="Whether to use a sample dataset")
    parser.add_argument("--satellite", type=str, choices=["landsat", "sentinel"], help="Satellite to use for training")
    parser.add_argument("--incl_bands", type=str, default="[1,2,3,4,5,6,7,8,9,10,11,12]", help="Bands to include, specified as a string of digits")
    parser.add_argument("--target_pos", type=int, default=-1, help="Position of the target band in the dataset (0-indexed)")
    parser.add_argument("--model_type", type=str, default="U_Net", help="Type of model to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--split", type=float, default=0.9, help="Train/Validation split")
    parser.add_argument("--early_stopping", type=int, default=-1, help="Number of epochs to wait before stopping training. -1 to disable.")

    parser.add_argument("--train_path", type=str, default="../data/training/", help="Path to the training data")
    parser.add_argument("--save_path", type=str, default="../models/", help="Path template for saving the model")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"], help="Device to use for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling the dataset")

    parser.add_argument("--note", type=str, default="", help="Note for model run context")
    
    # Added argument for binary mask training
    parser.add_argument("--binary_mask", action="store_true", help="Use a single output mask with sigmoid loss")

    # Parse the arguments
    args = parser.parse_args()

    # Process incl_bands to be a numpy array of integers, offset by -1
    args.incl_bands = np.array(eval(args.incl_bands)) - 1

    # Set device based on argument
    args.device = torch.device(args.device)
    print("\n" + args.note)
    print("\nTraining model with the following arguments:")
    print(vars(args))  # Print all arguments

    # Load data
    train_loader, valid_loader = load_data(args)

    # Train the model
    train_model(train_loader, valid_loader, args)


# Classes
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, paths, args):
        self.paths = paths
        self.target = args.target_pos
        self.incl_bands = args.incl_bands
        self.satellite = args.satellite
        self.binary_mask = args.binary_mask  # Store binary_mask flag

    def __getitem__(self, idx):
        """Get image and binary mask for a given index"""

        path = self.paths[idx]
        instance = np.load(path)

        # Get spectral bands
        bands = instance[:, :, self.incl_bands]  # Only include specified bands
        bands = bands.astype(np.float32) 

        # Normalise bands
        bands = utils.scale_bands(bands, self.satellite)

        # Convert to tensor
        bands = bands.transpose(2, 0, 1)
        bands = torch.tensor(bands)

        # Get target
        mask_1 = instance[:, :, self.target].astype(np.int8)  # Water = 1, Land = 0
        mask_1[np.where(mask_1 == -1)] = 0  # Set nodata values to 0

        if self.binary_mask:
            target = torch.tensor(mask_1, dtype=torch.float32).unsqueeze(0)  # Single channel
        else:
            mask_0 = 1 - mask_1
            target = torch.tensor(np.array([mask_0, mask_1]), dtype=torch.float32).squeeze()

        return bands, target

    def __len__(self):
        return len(self.paths)


# Functions
def load_data(args):
    """Load data from disk"""

    paths = glob.glob(args.train_path + "*.npy")
    print("Total images: {}".format(len(paths)))

    if args.sample:
        paths = paths[:100]

    # Shuffle the paths
    random.seed(args.seed)
    random.shuffle(paths)

    # Create datasets
    split = int(args.split * len(paths))
    train_data = TrainDataset(paths[:split], args)
    valid_data = TrainDataset(paths[split:], args)

    # Prepare data for PyTorch model
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size)

    return train_loader, valid_loader


def train_model(train_loader, valid_loader, args):
    # Define the model
    out_channels = 1 if args.binary_mask else 2  # Changed for binary classification
    if args.model_type == "U_Net":
        model = U_Net(len(args.incl_bands), out_channels)
    elif args.model_type == "R2U_Net":
        model = R2U_Net(len(args.incl_bands), out_channels)
    elif args.model_type == "AttU_Net":
        model = AttU_Net(len(args.incl_bands), out_channels)
    elif args.model_type == "R2AttU_Net":
        model = R2AttU_Net(len(args.incl_bands), out_channels)

    model.to(args.device)

    # Choose loss function based on binary mask flag
    criterion = nn.BCEWithLogitsLoss() if args.binary_mask else nn.CrossEntropyLoss()

    # Specify optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    min_loss = np.inf
    epochs_no_improve = 0

    for epoch in range(args.epochs):

        print("Epoch {} |".format(epoch + 1), end=" ")

        model.train()
        for images, target in train_loader:
            images, target = images.to(args.device), target.to(args.device)

            optimizer.zero_grad()
            output = model(images)

            if args.binary_mask:
                target = target.float()  # Ensure correct dtype for BCEWithLogitsLoss

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for images, target in valid_loader:
                images, target = images.to(args.device), target.to(args.device)
                output = model(images)

                if args.binary_mask:
                    target = target.float()

                valid_loss += criterion(output, target).item()

        valid_loss /= len(valid_loader)
        print("| Validation Loss: {}".format(round(valid_loss, 5)))

        if valid_loss < min_loss:
            torch.save(model.state_dict(), os.path.join(args.save_path, args.model_name + ".pth"))
            print("Model saved.")
            min_loss = valid_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if args.early_stopping > 0 and epochs_no_improve >= args.early_stopping:
            print("Early stopping triggered.")
            break


if __name__ == "__main__":
    main()
