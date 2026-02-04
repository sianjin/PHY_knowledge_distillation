#!/usr/bin/env python3
"""Quick script to inspect checkpoint without needing torch installed."""
import pickle
import sys

# Load the checkpoint file
with open('pkd_model.pt', 'rb') as f:
    checkpoint = pickle.load(f)

print("Checkpoint keys:", checkpoint.keys())
print("\n" + "="*60)

if 'model_state_dict' in checkpoint:
    print("\nModel state dict keys (first 20):")
    keys = list(checkpoint['model_state_dict'].keys())
    for key in keys[:20]:
        print(f"  {key}")
    if len(keys) > 20:
        print(f"  ... and {len(keys) - 20} more")

if 'model_config' in checkpoint:
    print("\nModel config:")
    for k, v in checkpoint['model_config'].items():
        print(f"  {k}: {v}")

if 'epoch' in checkpoint:
    print(f"\nTrained for {checkpoint['epoch']} epochs")

if 'train_loss' in checkpoint:
    print(f"Final train loss: {checkpoint['train_loss']:.4f}")

if 'val_loss' in checkpoint:
    print(f"Final val loss: {checkpoint['val_loss']:.4f}")

# Check a few parameter values to see if they look trained
if 'model_state_dict' in checkpoint:
    print("\n" + "="*60)
    print("Sample parameter values (to check if trained):")

    # Check mean head weights
    mean_weight_key = 'mean_head.net.0.weight'
    if mean_weight_key in checkpoint['model_state_dict']:
        mean_weights = checkpoint['model_state_dict'][mean_weight_key]
        print(f"\n{mean_weight_key}:")
        print(f"  Shape: {mean_weights.shape if hasattr(mean_weights, 'shape') else 'N/A'}")
        print(f"  First few values: {mean_weights.flatten()[:5] if hasattr(mean_weights, 'flatten') else 'N/A'}")

    # Check PACF head weights
    pacf_weight_key = 'pacf_head.net.0.weight'
    if pacf_weight_key in checkpoint['model_state_dict']:
        pacf_weights = checkpoint['model_state_dict'][pacf_weight_key]
        print(f"\n{pacf_weight_key}:")
        print(f"  Shape: {pacf_weights.shape if hasattr(pacf_weights, 'shape') else 'N/A'}")
        print(f"  First few values: {pacf_weights.flatten()[:5] if hasattr(pacf_weights, 'flatten') else 'N/A'}")
