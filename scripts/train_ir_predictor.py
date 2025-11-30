"""
Train IR predictor on FAIR-Play data.

Simple supervised learning:
- Input: Video features (512-d)
- Output: IR parameters (RT60, early_ratio, late_ratio)
- Loss: MSE on IR parameters
"""
import sys
sys.path.insert(0, '/home/seung')

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from learned_ir import SimpleIRPredictor


class IRDataset(Dataset):
    def __init__(self, json_path: str):
        with open(json_path) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        features = torch.FloatTensor(item['video_features'])
        params = item['ir_params']

        rt60 = torch.FloatTensor([params['rt60']])
        early_ratio = torch.FloatTensor([params['early_ratio']])
        late_ratio = torch.FloatTensor([params['late_ratio']])

        return features, rt60, early_ratio, late_ratio


def train_ir_predictor(
    dataset_path: str = 'ir_dataset_50.json',
    model_save_path: str = 'ir_predictor.pth',
    num_epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = 'cuda'
):
    """Train IR predictor."""

    # Load dataset
    dataset = IRDataset(dataset_path)

    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {train_size}, Val samples: {val_size}")

    # Model
    model = SimpleIRPredictor(input_size=512, hidden_size=256).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Loss
    criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0

        for features, rt60_gt, early_gt, late_gt in train_loader:
            features = features.to(device)
            rt60_gt = rt60_gt.to(device)
            early_gt = early_gt.to(device)
            late_gt = late_gt.to(device)

            # Forward
            rt60_pred, early_pred, late_pred = model(features)

            # Loss
            loss_rt60 = criterion(rt60_pred, rt60_gt.squeeze())
            loss_early = criterion(early_pred, early_gt.squeeze())
            loss_late = criterion(late_pred, late_gt.squeeze())

            loss = loss_rt60 + loss_early + loss_late

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for features, rt60_gt, early_gt, late_gt in val_loader:
                features = features.to(device)
                rt60_gt = rt60_gt.to(device)
                early_gt = early_gt.to(device)
                late_gt = late_gt.to(device)

                rt60_pred, early_pred, late_pred = model(features)

                loss_rt60 = criterion(rt60_pred, rt60_gt.squeeze())
                loss_early = criterion(early_pred, early_gt.squeeze())
                loss_late = criterion(late_pred, late_gt.squeeze())

                loss = loss_rt60 + loss_early + loss_late
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to {model_save_path}")

    return model


def analyze_learned_params(dataset_path: str = 'ir_dataset_50.json'):
    """Analyze IR parameters extracted from GT."""

    with open(dataset_path) as f:
        data = json.load(f)

    rt60_values = [item['ir_params']['rt60'] for item in data]
    early_values = [item['ir_params']['early_ratio'] for item in data]
    late_values = [item['ir_params']['late_ratio'] for item in data]

    print("="*60)
    print("IR PARAMETER STATISTICS (from GT binaural)")
    print("="*60)

    print(f"\nRT60 (seconds):")
    print(f"  Mean: {np.mean(rt60_values):.3f}")
    print(f"  Std:  {np.std(rt60_values):.3f}")
    print(f"  Min:  {np.min(rt60_values):.3f}")
    print(f"  Max:  {np.max(rt60_values):.3f}")

    print(f"\nEarly Ratio:")
    print(f"  Mean: {np.mean(early_values):.3f}")
    print(f"  Std:  {np.std(early_values):.3f}")
    print(f"  Min:  {np.min(early_values):.3f}")
    print(f"  Max:  {np.max(early_values):.3f}")

    print(f"\nLate Ratio:")
    print(f"  Mean: {np.mean(late_values):.3f}")
    print(f"  Std:  {np.std(late_values):.3f}")
    print(f"  Min:  {np.min(late_values):.3f}")
    print(f"  Max:  {np.max(late_values):.3f}")

    print("\n" + "="*60)

    # Compare with Schroeder defaults
    print("\nComparison with Schroeder IR (RT60=0.6s):")
    print(f"  GT RT60 mean: {np.mean(rt60_values):.3f}s")
    print(f"  Schroeder:    0.600s")
    print(f"  → Mismatch: {abs(np.mean(rt60_values) - 0.6):.3f}s")

    if np.mean(rt60_values) < 0.3:
        print("\n⚠️  GT data has MUCH SHORTER RT60 than Schroeder!")
        print("   This explains why Schroeder IR degrades performance.")
        print("   FAIR-Play likely uses anechoic or dry acoustics.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ir_dataset_50.json')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--analyze_only', action='store_true')
    args = parser.parse_args()

    # First, analyze parameters
    analyze_learned_params(args.dataset)

    if not args.analyze_only:
        # Train model
        print("\nTraining IR predictor...\n")
        model = train_ir_predictor(
            dataset_path=args.dataset,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device
        )
