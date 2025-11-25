import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from preprocessing import normalize_zscore, extract_iq_channels


class RadioDataset(Dataset):
    """
    Fast PyTorch Dataset for loading RF signals from HDF5 files.
    Applies minimal preprocessing for speed.
    """
    def __init__(self, clean_path, jammed_path, window_size=1024, split='train', 
                 normalization='zscore', use_psd=False, max_samples=None):
        self.window_size = window_size
        self.use_psd = use_psd
        
        print(f"Loading {split} data...")

        # Calculate how many samples to load if max_samples is specified
        samples_to_load = None
        if max_samples is not None:
            samples_to_load = max_samples // 2  # Half from each class

        # Load data from HDF5 files - use slicing to limit memory usage
        with h5py.File(clean_path, 'r') as f:
            if samples_to_load is not None:
                clean_signals = np.array(f[split]['signals'][:samples_to_load])
                clean_labels = np.array(f[split]['jammed'][:samples_to_load])
            else:
                clean_signals = np.array(f[split]['signals'])
                clean_labels = np.array(f[split]['jammed'])

        with h5py.File(jammed_path, 'r') as f:
            if samples_to_load is not None:
                jammed_signals = np.array(f[split]['signals'][:samples_to_load])
                jammed_labels = np.array(f[split]['jammed'][:samples_to_load])
            else:
                jammed_signals = np.array(f[split]['signals'])
                jammed_labels = np.array(f[split]['jammed'])

        print(f"Clean signals: {clean_signals.shape}, Jammed signals: {jammed_signals.shape}")

        # Combine and convert to correct types
        all_signals = np.concatenate([clean_signals, jammed_signals], axis=0).astype(np.float32)
        # Use the actual labels from the files (they correctly identify jammed vs clean)
        all_labels = np.concatenate([clean_labels.astype(int), jammed_labels.astype(int)], axis=0)
        
        # Simple preprocessing - treat as I/Q data
        if all_signals.shape[1] >= window_size * 2:
            # Assume interleaved I/Q format: [I1, Q1, I2, Q2, ...]
            signals_reshaped = all_signals[:, :window_size*2].reshape(-1, 2, window_size)
        else:
            # Treat as single channel, duplicate for I/Q
            I_channel = all_signals[:, :window_size]
            Q_channel = np.zeros_like(I_channel)  # Zero Q channel
            signals_reshaped = np.stack([I_channel, Q_channel], axis=1)
        
        # Fast batch normalization
        if normalization == 'zscore':
            mean = signals_reshaped.mean(axis=(0, 2), keepdims=True)
            std = signals_reshaped.std(axis=(0, 2), keepdims=True) + 1e-8
            signals_reshaped = (signals_reshaped - mean) / std
        
        # Store as tensors for fast access
        self.data = torch.from_numpy(signals_reshaped).float()  # Shape: (N, 2, window_size)
        self.labels = torch.from_numpy(all_labels).long()
        
        print(f"Dataset initialized with {len(self.data)} samples, shape: {self.data.shape}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_dataloaders(clean_path, jammed_path, window_size=1024, batch_size=32, 
                   val_split=0.2, num_workers=4, normalization='zscore', use_psd=False, max_samples=None):
    """
    Create train and validation data loaders with optimized settings.
    """
    # Create datasets for train split
    train_dataset = RadioDataset(
        clean_path, jammed_path, 
        window_size=window_size, 
        split='train',
        normalization=normalization,
        use_psd=use_psd,
        max_samples=max_samples
    )
    
    # Create validation dataset from test split
    val_dataset = RadioDataset(
        clean_path, jammed_path,
        window_size=window_size,
        split='test',
        normalization=normalization, 
        use_psd=use_psd,
        max_samples=max_samples  # Also limit validation set
    )
    
    # Optimized data loaders for speed
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=min(num_workers, 4),  # Limit workers to prevent overhead
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=min(num_workers, 2),  # Fewer workers for validation
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader


def get_input_size(window_size=1024, use_psd=False):
    """
    Get the input size for models based on preprocessing configuration.
    """
    if use_psd:
        return 3 * window_size  # I, Q, and PSD channels
    else:
        return 2 * window_size  # I and Q channels only
