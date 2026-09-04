"""Training loop for PKD model.

Training with Gaussian innovations:
------------------------------------
The supported PKD model predicts a configuration-dependent standard deviation
sigma_t and models the innovation as eps_t ~ N(0, sigma_t^2).

Training minimizes the Gaussian negative log-likelihood, where
eps_t = X_t - mu_t is the innovation and mu_t is the AR mean.

Legacy experimental innovation implementations remain loadable for old
checkpoints, but they are not used by the training or evaluation pipeline.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm


class PKDDataset(Dataset):
    """Dataset for teacher-generated sequences."""

    def __init__(self, sequences, configs, ar_order=10, eps=1e-12):
        """
        Args:
            sequences: list of (T_n,) arrays of gamma_eff from teacher
            configs: list of config dicts for each sequence
            ar_order: AR order (for history)
            eps: minimum value to prevent log(0) -> -inf
        """
        self.configs = configs
        self.ar_order = ar_order

        # Data is already in natural log scale (converted in data_loader.py)
        # Store uniform sequences contiguously so DataLoader can fetch a whole
        # batch with NumPy indexing instead of constructing samples one by one.
        sequence_lengths = np.asarray([len(sequence) for sequence in sequences])
        self.uniform_sequences = (
            len(sequence_lengths) > 0
            and np.all(sequence_lengths == sequence_lengths[0])
        )
        if self.uniform_sequences:
            self.X_sequences = np.asarray(sequences, dtype=np.float32)
            self.sequence_length = int(sequence_lengths[0])
            self.samples_per_sequence = self.sequence_length - ar_order
        else:
            self.X_sequences = [
                np.asarray(sequence, dtype=np.float32) for sequence in sequences
            ]
            self.sample_offsets = np.concatenate((
                [0],
                np.cumsum(np.maximum(sequence_lengths - ar_order, 0)),
            ))

        self.static_configs = all(isinstance(config, dict) for config in configs)
        if self.static_configs and configs:
            self.config_arrays = {}
            for key in configs[0]:
                values = [config[key] for config in configs]
                dtype = np.float32 if isinstance(values[0], float) else np.int64
                self.config_arrays[key] = np.asarray(values, dtype=dtype)

        # Check for any non-finite values
        for i, X_seq in enumerate(self.X_sequences):
            if not np.all(np.isfinite(X_seq)):
                print(f"Warning: Sequence {i} contains non-finite values after log transform")
                print(f"  Min: {np.min(sequences[i])}, Max: {np.max(sequences[i])}")
                print(f"  Finite ratio: {np.isfinite(X_seq).mean():.4f}")

    def __len__(self):
        if self.uniform_sequences:
            return len(self.X_sequences) * self.samples_per_sequence
        return int(self.sample_offsets[-1])

    def __getitem__(self, idx):
        if self.uniform_sequences:
            seq_idx, time_offset = divmod(idx, self.samples_per_sequence)
            t = time_offset + self.ar_order
        else:
            seq_idx = np.searchsorted(self.sample_offsets, idx, side='right') - 1
            t = idx - self.sample_offsets[seq_idx] + self.ar_order

        # Get configuration at time t
        config_t = self.configs[seq_idx][t] if isinstance(self.configs[seq_idx], list) else self.configs[seq_idx]

        return {
            'X_t': torch.tensor(self.X_sequences[seq_idx][t], dtype=torch.float32),
            'X_hist': torch.from_numpy(
                self.X_sequences[seq_idx][t-self.ar_order:t][::-1].copy()
            ),
            'config': config_t
        }

    def __getitems__(self, indices):
        """Fetch uniform, static-config batches without per-sample Python work."""
        if not (self.uniform_sequences and self.static_configs):
            return [self[index] for index in indices]

        indices = np.asarray(indices, dtype=np.int64)
        seq_indices, time_offsets = np.divmod(indices, self.samples_per_sequence)
        time_indices = time_offsets + self.ar_order
        history_indices = time_indices[:, None] - np.arange(
            1, self.ar_order + 1, dtype=np.int64
        )

        X_t = torch.from_numpy(self.X_sequences[seq_indices, time_indices])
        X_hist = torch.from_numpy(self.X_sequences[seq_indices[:, None], history_indices])
        config_dict = {
            key: torch.from_numpy(values[seq_indices])
            for key, values in self.config_arrays.items()
        }
        return X_t, X_hist, config_dict


def collate_fn(batch):
    """Collate batch of samples."""
    if (
        isinstance(batch, tuple)
        and len(batch) == 3
        and torch.is_tensor(batch[0])
    ):
        return batch

    X_t = torch.stack([b['X_t'] for b in batch])
    X_hist = torch.stack([b['X_hist'] for b in batch])

    # Collate configs
    config_dict = {}
    config_keys = batch[0]['config'].keys()
    for key in config_keys:
        values = [b['config'][key] for b in batch]
        if isinstance(values[0], (int, float)):
            config_dict[key] = torch.tensor(values)
        else:
            config_dict[key] = torch.stack(values) if torch.is_tensor(values[0]) else torch.tensor(values)

    return X_t, X_hist, config_dict


def train_epoch(model, dataloader, optimizer, device, clip_grad=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(dataloader, desc="Training", mininterval=2.0)
    for X_t, X_hist, config_dict in pbar:
        # Move to device
        X_t = X_t.to(device, non_blocking=True)
        X_hist = X_hist.to(device, non_blocking=True)
        config_dict = {
            k: v.to(device, non_blocking=True) for k, v in config_dict.items()
        }

        # Forward
        optimizer.zero_grad(set_to_none=True)
        log_q, info = model.compute_log_likelihood(X_t, X_hist, config_dict)

        # Loss: negative log-likelihood
        loss = -log_q.mean()

        # Check for non-finite values (NaN or Inf) and print diagnostics
        if not torch.isfinite(loss):
            print(f"\n{'='*60}")
            print(f"Non-finite loss detected: {loss.item()}")
            print(f"{'='*60}")
            print(f"\nInput data:")
            print(f"  X_t: min={X_t.min():.4f}, max={X_t.max():.4f}, mean={X_t.mean():.4f}")
            print(f"  X_t finite ratio: {torch.isfinite(X_t).float().mean().item():.4f}")
            print(f"  X_hist: min={X_hist.min():.4f}, max={X_hist.max():.4f}, mean={X_hist.mean():.4f}")
            print(f"  X_hist finite ratio: {torch.isfinite(X_hist).float().mean().item():.4f}")

            print(f"\nModel predictions:")
            print(f"  mu: min={info['mu'].min():.4f}, max={info['mu'].max():.4f}, mean={info['mu'].mean():.4f}")
            print(f"  mu finite ratio: {torch.isfinite(info['mu']).float().mean().item():.4f}")
            print(f"  eps: min={info['eps'].min():.4f}, max={info['eps'].max():.4f}, mean={info['eps'].mean():.4f}")
            print(f"  eps finite ratio: {torch.isfinite(info['eps']).float().mean().item():.4f}")

            # Check for standardized innovation (only exists for Gaussian)
            if 'z' in info:
                print(f"  z: min={info['z'].min():.4f}, max={info['z'].max():.4f}, mean={info['z'].mean():.4f}")
                print(f"  z finite ratio: {torch.isfinite(info['z']).float().mean().item():.4f}")

            print(f"\nAR coefficients:")
            print(f"  phi: min={info['phi'].min():.4f}, max={info['phi'].max():.4f}, mean={info['phi'].mean():.4f}")
            phi_sum = info['phi'].sum(dim=-1)
            print(f"  phi sum: min={phi_sum.min():.4f}, max={phi_sum.max():.4f}, mean={phi_sum.mean():.4f}")
            print(f"  |phi| sum: min={info['phi'].abs().sum(dim=-1).min():.4f}, max={info['phi'].abs().sum(dim=-1).max():.4f}")
            print(f"  kappa: min={info['kappa'].min():.4f}, max={info['kappa'].max():.4f}, mean={info['kappa'].mean():.4f}")
            print(f"  |kappa| max: {info['kappa'].abs().max():.4f} (should be < 0.95 for stability)")

            print(f"\nInnovation parameters:")
            innov = info['innov_params']
            if isinstance(innov, dict):  # SGN (sigma, beta, lambda)
                print(f"  sigma: min={innov['sigma'].min():.4f}, max={innov['sigma'].max():.4f}, mean={innov['sigma'].mean():.4f}")
                print(f"  beta: min={innov['beta'].min():.4f}, max={innov['beta'].max():.4f}, mean={innov['beta'].mean():.4f}")
                print(f"  lambda: min={innov['lambda'].min():.4f}, max={innov['lambda'].max():.4f}, mean={innov['lambda'].mean():.4f}")
            elif innov.ndim == 1:  # Gaussian (sigma)
                print(f"  sigma: min={innov.min():.4f}, max={innov.max():.4f}, mean={innov.mean():.4f}")
            else:  # Flow (psi)
                print(f"  psi shape: {innov.shape}")

            print(f"\nLog-likelihood:")
            print(f"  log_q: min={log_q.min():.4f}, max={log_q.max():.4f}, mean={log_q.mean():.4f}")
            print(f"  log_q finite ratio: {torch.isfinite(log_q).float().mean().item():.4f}")
            print(f"  log_q non-finite count: {(~torch.isfinite(log_q)).sum()} / {log_q.numel()}")

            raise ValueError("Non-finite loss detected")

        # Backward
        loss.backward()

        # Gradient clipping
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        # Track
        batch_size = X_t.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        pbar.set_postfix({'loss': loss.item()})

    return total_loss / total_samples


def validate(model, dataloader, device):
    """Validation."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    non_finite_batches = 0

    with torch.no_grad():
        for X_t, X_hist, config_dict in dataloader:
            X_t = X_t.to(device, non_blocking=True)
            X_hist = X_hist.to(device, non_blocking=True)
            config_dict = {
                k: v.to(device, non_blocking=True) for k, v in config_dict.items()
            }

            log_q, _ = model.compute_log_likelihood(X_t, X_hist, config_dict)
            loss = -log_q.mean()

            # Check for non-finite values in validation
            if not torch.isfinite(loss):
                non_finite_batches += 1
                continue  # Skip this batch

            batch_size = X_t.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    if non_finite_batches > 0:
        print(f"  Warning: {non_finite_batches} validation batches had non-finite loss")

    return total_loss / total_samples if total_samples > 0 else float('inf')


def train_pkd(model, train_sequences, train_configs,
              val_sequences, val_configs,
              num_epochs=100,
              batch_size=256,
              lr=1e-3,
              device='cuda',
              save_path='pkd_model.pt',
              early_stopping_patience=5,
              model_config=None):
    """
    Complete training pipeline.

    Args:
        model: PKDModel instance
        train_sequences: list of teacher gamma_eff sequences
        train_configs: list of config dicts
        val_sequences: validation sequences
        val_configs: validation configs
        num_epochs: number of training epochs
        batch_size: batch size
        lr: learning rate
        device: device
        save_path: path to save best model
        early_stopping_patience: epochs to wait before early stopping
        model_config: dict of model configuration (saved with checkpoint)
    """
    # Create datasets
    train_dataset = PKDDataset(train_sequences, train_configs, model.ar_order)
    val_dataset = PKDDataset(val_sequences, val_configs, model.ar_order)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, collate_fn=collate_fn,
                             num_workers=8, pin_memory=True,
                             persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, collate_fn=collate_fn,
                           num_workers=4, pin_memory=True,
                           persistent_workers=True, prefetch_factor=2)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Move model to device
    model = model.to(device)

    # Training loop
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        val_loss = validate(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f}")

        # Log relative difference if val < train
        if val_loss < train_loss:
            rel_diff = (train_loss - val_loss) / train_loss * 100
            print(f"  Note: Val loss is {rel_diff:.1f}% lower than train loss (may be normal with dropout/regularization)")

        # Step scheduler and check if LR changed
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        if new_lr < old_lr:
            print(f"  Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")

        # Save best model and check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            if model_config is not None:
                checkpoint['model_config'] = model_config
            torch.save(checkpoint, save_path)
            print(f"Saved best model with val loss {val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")

            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break

    # Load best model before returning
    print(f"\nLoading best model from {save_path}")
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model
