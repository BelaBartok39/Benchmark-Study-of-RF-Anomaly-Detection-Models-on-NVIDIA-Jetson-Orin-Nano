"""
Data preprocessing utilities for RF signal processing.

This module provides functions for I/Q channel extraction, PSD computation,
and input validation for RF anomaly detection.
"""

import numpy as np
import torch
from scipy import signal
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


def extract_iq_channels(rf_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Extract I/Q channels from complex RF signal data.
    
    Args:
        rf_data: Complex RF signal data of shape (batch_size, sequence_length) or (sequence_length,)
        
    Returns:
        torch.Tensor: I/Q channels stacked as [I, Q] with shape (batch_size, 2, sequence_length)
                     or (2, sequence_length) for single sample
        
    Raises:
        ValueError: If input data is not complex or has invalid shape
        TypeError: If input data type is not supported
    """
    # Input validation
    if not isinstance(rf_data, (np.ndarray, torch.Tensor)):
        raise TypeError(f"Input must be numpy array or torch tensor, got {type(rf_data)}")
    
    # Convert to numpy if torch tensor
    if isinstance(rf_data, torch.Tensor):
        rf_data_np = rf_data.detach().cpu().numpy()
    else:
        rf_data_np = rf_data
    
    # Check if data is complex
    if not np.iscomplexobj(rf_data_np):
        raise ValueError("Input data must be complex (I/Q signal)")
    
    # Handle different input shapes
    if rf_data_np.ndim == 1:
        # Single sample: (sequence_length,)
        i_channel = np.real(rf_data_np)
        q_channel = np.imag(rf_data_np)
        iq_stacked = np.stack([i_channel, q_channel], axis=0)  # Shape: (2, sequence_length)
    elif rf_data_np.ndim == 2:
        # Batch: (batch_size, sequence_length)
        i_channel = np.real(rf_data_np)
        q_channel = np.imag(rf_data_np)
        iq_stacked = np.stack([i_channel, q_channel], axis=1)  # Shape: (batch_size, 2, sequence_length)
    else:
        raise ValueError(f"Input data must be 1D or 2D, got shape {rf_data_np.shape}")
    
    # Convert to torch tensor
    return torch.from_numpy(iq_stacked).float()


def compute_psd(rf_signal: Union[np.ndarray, torch.Tensor], 
                nperseg: int = 256,
                noverlap: Optional[int] = None,
                nfft: Optional[int] = None,
                fs: float = 1.0,
                window: str = 'hann',
                scaling: str = 'density',
                return_onesided: bool = True) -> torch.Tensor:
    """
    Compute Power Spectral Density (PSD) of RF signal using Welch's method.
    
    Args:
        rf_signal: Complex RF signal data of shape (batch_size, sequence_length) or (sequence_length,)
        nperseg: Length of each segment for PSD computation
        noverlap: Number of points to overlap between segments
        nfft: Length of the FFT used
        fs: Sampling frequency
        window: Desired window to use
        scaling: Return power spectral density ('density') or power spectrum ('spectrum')
        return_onesided: If True, return one-sided spectrum for real signals, two-sided for complex
        
    Returns:
        torch.Tensor: PSD values with shape (batch_size, psd_bins) or (psd_bins,)
                     For complex signals: psd_bins = nfft (two-sided) or nfft//2+1 (one-sided)
        
    Raises:
        ValueError: If input parameters are invalid
        TypeError: If input data type is not supported
    """
    # Input validation
    if not isinstance(rf_signal, (np.ndarray, torch.Tensor)):
        raise TypeError(f"Input must be numpy array or torch tensor, got {type(rf_signal)}")
    
    if nperseg <= 0:
        raise ValueError(f"nperseg must be positive, got {nperseg}")
    
    # Convert to numpy if torch tensor
    if isinstance(rf_signal, torch.Tensor):
        signal_np = rf_signal.detach().cpu().numpy()
    else:
        signal_np = rf_signal
    
    # Adjust nperseg for short signals
    if signal_np.ndim == 1:
        signal_len = len(signal_np)
    else:
        signal_len = signal_np.shape[1]
    
    if nperseg > signal_len:
        nperseg = signal_len
        logger.warning(f"nperseg={nperseg} is greater than signal length {signal_len}, using nperseg = {signal_len}")
    
    # Set default overlap
    if noverlap is None:
        noverlap = nperseg // 2
    
    # Ensure noverlap is valid
    if noverlap >= nperseg:
        noverlap = nperseg // 2
    
    # Set default nfft (scipy.signal.welch uses nperseg as default)
    if nfft is None:
        nfft = nperseg
    
    try:
        if signal_np.ndim == 1:
            # Single sample
            freqs, psd = signal.welch(signal_np, fs=fs, window=window, 
                                    nperseg=nperseg, noverlap=noverlap, 
                                    nfft=nfft, scaling=scaling,
                                    return_onesided=return_onesided)
            psd_tensor = torch.from_numpy(psd).float()
        elif signal_np.ndim == 2:
            # Batch processing
            batch_size, seq_len = signal_np.shape
            psd_list = []
            
            for i in range(batch_size):
                freqs, psd = signal.welch(signal_np[i], fs=fs, window=window,
                                        nperseg=nperseg, noverlap=noverlap,
                                        nfft=nfft, scaling=scaling,
                                        return_onesided=return_onesided)
                psd_list.append(psd)
            
            psd_tensor = torch.from_numpy(np.stack(psd_list, axis=0)).float()
        else:
            raise ValueError(f"Input signal must be 1D or 2D, got shape {signal_np.shape}")
            
    except Exception as e:
        logger.error(f"PSD computation failed: {e}")
        raise ValueError(f"PSD computation failed: {e}")
    
    return psd_tensor


def validate_rf_input(rf_data: Union[np.ndarray, torch.Tensor], 
                     expected_shape: Optional[Tuple[int, ...]] = None,
                     check_finite: bool = True,
                     check_complex: bool = True) -> bool:
    """
    Validate RF signal input data for corruption and format issues.
    
    Args:
        rf_data: Input RF signal data
        expected_shape: Expected shape tuple (optional)
        check_finite: Whether to check for finite values
        check_complex: Whether to check for complex data type
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ValueError: If validation fails
        TypeError: If input type is invalid
    """
    if not isinstance(rf_data, (np.ndarray, torch.Tensor)):
        raise TypeError(f"Input must be numpy array or torch tensor, got {type(rf_data)}")
    
    # Convert to numpy for validation
    if isinstance(rf_data, torch.Tensor):
        data_np = rf_data.detach().cpu().numpy()
    else:
        data_np = rf_data
    
    # Check shape
    if expected_shape is not None and data_np.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {data_np.shape}")
    
    # Check for empty data
    if data_np.size == 0:
        raise ValueError("Input data is empty")
    
    # Check for finite values
    if check_finite and not np.all(np.isfinite(data_np)):
        raise ValueError("Input data contains non-finite values (NaN or Inf)")
    
    # Check for complex data type
    if check_complex and not np.iscomplexobj(data_np):
        raise ValueError("Input data must be complex for RF signal processing")
    
    # Check for zero variance (corrupted/constant signal)
    if np.var(data_np) == 0:
        logger.warning("Input signal has zero variance (constant signal)")
    
    return True


def handle_corrupted_samples(rf_data: Union[np.ndarray, torch.Tensor],
                           labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
                           corruption_threshold: float = 1e-10) -> Tuple[torch.Tensor, Optional[torch.Tensor], list]:
    """
    Identify and handle corrupted RF signal samples.
    
    Args:
        rf_data: RF signal data of shape (batch_size, sequence_length)
        labels: Optional labels corresponding to the data
        corruption_threshold: Minimum variance threshold to consider signal valid
        
    Returns:
        Tuple containing:
        - Clean RF data with corrupted samples removed
        - Clean labels (if provided) with corresponding samples removed
        - List of indices of corrupted samples
        
    Raises:
        ValueError: If all samples are corrupted
    """
    # Convert to numpy for processing
    if isinstance(rf_data, torch.Tensor):
        data_np = rf_data.detach().cpu().numpy()
    else:
        data_np = rf_data.copy()
    
    if labels is not None:
        if isinstance(labels, torch.Tensor):
            labels_np = labels.detach().cpu().numpy()
        else:
            labels_np = labels.copy()
    
    corrupted_indices = []
    
    # Check each sample
    for i in range(data_np.shape[0]):
        sample = data_np[i]
        
        # Check for non-finite values
        if not np.all(np.isfinite(sample)):
            corrupted_indices.append(i)
            continue
            
        # Check for zero/low variance
        if np.var(sample) < corruption_threshold:
            corrupted_indices.append(i)
            continue
    
    if len(corrupted_indices) == data_np.shape[0]:
        raise ValueError("All samples are corrupted")
    
    # Remove corrupted samples
    valid_indices = [i for i in range(data_np.shape[0]) if i not in corrupted_indices]
    clean_data = torch.from_numpy(data_np[valid_indices])
    # Keep complex data as complex, convert only if originally real
    if not np.iscomplexobj(data_np):
        clean_data = clean_data.float()
    
    clean_labels = None
    if labels is not None:
        clean_labels = torch.from_numpy(labels_np[valid_indices])
    
    if corrupted_indices:
        logger.warning(f"Removed {len(corrupted_indices)} corrupted samples out of {data_np.shape[0]}")
    
    return clean_data, clean_labels, corrupted_indices

def normalize_zscore(data: Union[np.ndarray, torch.Tensor], 
                    axis: Optional[int] = None,
                    epsilon: float = 1e-8) -> torch.Tensor:
    """
    Apply z-score normalization to data.
    
    Args:
        data: Input data to normalize
        axis: Axis along which to compute statistics (None for global stats)
        epsilon: Small value to prevent division by zero
        
    Returns:
        torch.Tensor: Z-score normalized data
        
    Raises:
        ValueError: If data has zero variance
    """
    # Convert to torch tensor
    if isinstance(data, np.ndarray):
        data_tensor = torch.from_numpy(data).float()
    else:
        data_tensor = data.float()
    
    # Compute mean and std
    if axis is None:
        mean = torch.mean(data_tensor)
        std = torch.std(data_tensor)
    else:
        mean = torch.mean(data_tensor, dim=axis, keepdim=True)
        std = torch.std(data_tensor, dim=axis, keepdim=True)
    
    # Check for zero variance
    if torch.any(std < epsilon):
        logger.warning("Data has near-zero variance, normalization may be unstable")
        std = torch.clamp(std, min=epsilon)
    
    # Apply z-score normalization
    normalized = (data_tensor - mean) / std
    
    return normalized


def normalize_minmax(data: Union[np.ndarray, torch.Tensor],
                    feature_range: Tuple[float, float] = (0.0, 1.0),
                    axis: Optional[int] = None,
                    epsilon: float = 1e-8) -> torch.Tensor:
    """
    Apply min-max normalization to data.
    
    Args:
        data: Input data to normalize
        feature_range: Target range for normalized data (min, max)
        axis: Axis along which to compute statistics (None for global stats)
        epsilon: Small value to prevent division by zero
        
    Returns:
        torch.Tensor: Min-max normalized data
        
    Raises:
        ValueError: If feature_range is invalid
    """
    if feature_range[0] >= feature_range[1]:
        raise ValueError(f"Invalid feature_range: {feature_range}. Min must be less than max.")
    
    # Convert to torch tensor
    if isinstance(data, np.ndarray):
        data_tensor = torch.from_numpy(data).float()
    else:
        data_tensor = data.float()
    
    # Compute min and max
    if axis is None:
        data_min = torch.min(data_tensor)
        data_max = torch.max(data_tensor)
    else:
        data_min = torch.min(data_tensor, dim=axis, keepdim=True)[0]
        data_max = torch.max(data_tensor, dim=axis, keepdim=True)[0]
    
    # Check for zero range
    data_range = data_max - data_min
    if torch.any(data_range < epsilon):
        logger.warning("Data has near-zero range, normalization may be unstable")
        data_range = torch.clamp(data_range, min=epsilon)
    
    # Apply min-max normalization
    normalized = (data_tensor - data_min) / data_range
    
    # Scale to target range
    target_min, target_max = feature_range
    normalized = normalized * (target_max - target_min) + target_min
    
    return normalized


def robust_normalize(data: Union[np.ndarray, torch.Tensor],
                    method: str = 'median_mad',
                    axis: Optional[int] = None,
                    epsilon: float = 1e-8) -> torch.Tensor:
    """
    Apply robust normalization using statistics less sensitive to outliers.
    
    Args:
        data: Input data to normalize
        method: Robust normalization method ('median_mad', 'iqr', 'quantile')
        axis: Axis along which to compute statistics (None for global stats)
        epsilon: Small value to prevent division by zero
        
    Returns:
        torch.Tensor: Robust normalized data
        
    Raises:
        ValueError: If method is not supported
    """
    if method not in ['median_mad', 'iqr', 'quantile']:
        raise ValueError(f"Unsupported method: {method}. Choose from 'median_mad', 'iqr', 'quantile'")
    
    # Convert to torch tensor
    if isinstance(data, np.ndarray):
        data_tensor = torch.from_numpy(data).float()
    else:
        data_tensor = data.float()
    
    if method == 'median_mad':
        # Median Absolute Deviation normalization
        if axis is None:
            median = torch.median(data_tensor)
            mad = torch.median(torch.abs(data_tensor - median))
        else:
            median = torch.median(data_tensor, dim=axis, keepdim=True)[0]
            mad = torch.median(torch.abs(data_tensor - median), dim=axis, keepdim=True)[0]
        
        mad = torch.clamp(mad, min=epsilon)
        normalized = (data_tensor - median) / mad
        
    elif method == 'iqr':
        # Interquartile Range normalization
        if axis is None:
            q25 = torch.quantile(data_tensor, 0.25)
            q75 = torch.quantile(data_tensor, 0.75)
            median = torch.median(data_tensor)
        else:
            q25 = torch.quantile(data_tensor, 0.25, dim=axis, keepdim=True)
            q75 = torch.quantile(data_tensor, 0.75, dim=axis, keepdim=True)
            median = torch.median(data_tensor, dim=axis, keepdim=True)[0]
        
        iqr = q75 - q25
        iqr = torch.clamp(iqr, min=epsilon)
        normalized = (data_tensor - median) / iqr
        
    elif method == 'quantile':
        # Quantile normalization (0.1 to 0.9 range)
        if axis is None:
            q10 = torch.quantile(data_tensor, 0.1)
            q90 = torch.quantile(data_tensor, 0.9)
        else:
            q10 = torch.quantile(data_tensor, 0.1, dim=axis, keepdim=True)
            q90 = torch.quantile(data_tensor, 0.9, dim=axis, keepdim=True)
        
        q_range = q90 - q10
        q_range = torch.clamp(q_range, min=epsilon)
        normalized = (data_tensor - q10) / q_range
    
    return normalized


def detect_outliers(data: Union[np.ndarray, torch.Tensor],
                   method: str = 'iqr',
                   threshold: float = 1.5,
                   axis: Optional[int] = None) -> torch.Tensor:
    """
    Detect outliers in data using robust statistical methods.
    
    Args:
        data: Input data to analyze
        method: Outlier detection method ('iqr', 'mad', 'zscore')
        threshold: Threshold multiplier for outlier detection
        axis: Axis along which to compute statistics (None for global stats)
        
    Returns:
        torch.Tensor: Boolean mask where True indicates outliers
        
    Raises:
        ValueError: If method is not supported
    """
    if method not in ['iqr', 'mad', 'zscore']:
        raise ValueError(f"Unsupported method: {method}. Choose from 'iqr', 'mad', 'zscore'")
    
    # Convert to torch tensor
    if isinstance(data, np.ndarray):
        data_tensor = torch.from_numpy(data).float()
    else:
        data_tensor = data.float()
    
    if method == 'iqr':
        # IQR-based outlier detection
        if axis is None:
            q25 = torch.quantile(data_tensor, 0.25)
            q75 = torch.quantile(data_tensor, 0.75)
        else:
            q25 = torch.quantile(data_tensor, 0.25, dim=axis, keepdim=True)
            q75 = torch.quantile(data_tensor, 0.75, dim=axis, keepdim=True)
        
        iqr = q75 - q25
        lower_bound = q25 - threshold * iqr
        upper_bound = q75 + threshold * iqr
        outliers = (data_tensor < lower_bound) | (data_tensor > upper_bound)
        
    elif method == 'mad':
        # MAD-based outlier detection
        if axis is None:
            median = torch.median(data_tensor)
            mad = torch.median(torch.abs(data_tensor - median))
        else:
            median = torch.median(data_tensor, dim=axis, keepdim=True)[0]
            mad = torch.median(torch.abs(data_tensor - median), dim=axis, keepdim=True)[0]
        
        # Modified z-score using MAD
        modified_z_score = 0.6745 * (data_tensor - median) / mad
        outliers = torch.abs(modified_z_score) > threshold
        
    elif method == 'zscore':
        # Z-score based outlier detection
        if axis is None:
            mean = torch.mean(data_tensor)
            std = torch.std(data_tensor)
        else:
            mean = torch.mean(data_tensor, dim=axis, keepdim=True)
            std = torch.std(data_tensor, dim=axis, keepdim=True)
        
        z_scores = torch.abs((data_tensor - mean) / std)
        outliers = z_scores > threshold
    
    return outliers


def augment_rf_signal(rf_signal: Union[np.ndarray, torch.Tensor],
                     augmentation_type: str = 'noise',
                     **kwargs) -> torch.Tensor:
    """
    Apply data augmentation techniques to RF signals.
    
    Args:
        rf_signal: Complex RF signal data
        augmentation_type: Type of augmentation ('noise', 'phase_shift', 'amplitude_scale', 'frequency_shift')
        **kwargs: Additional parameters for specific augmentation types
        
    Returns:
        torch.Tensor: Augmented RF signal
        
    Raises:
        ValueError: If augmentation_type is not supported
    """
    supported_types = ['noise', 'phase_shift', 'amplitude_scale', 'frequency_shift']
    if augmentation_type not in supported_types:
        raise ValueError(f"Unsupported augmentation_type: {augmentation_type}. Choose from {supported_types}")
    
    # Convert to torch tensor
    if isinstance(rf_signal, np.ndarray):
        signal_tensor = torch.from_numpy(rf_signal)
    else:
        signal_tensor = rf_signal.clone()
    
    if augmentation_type == 'noise':
        # Add Gaussian noise
        noise_std = kwargs.get('noise_std', 0.01)
        noise_real = torch.randn_like(signal_tensor.real) * noise_std
        noise_imag = torch.randn_like(signal_tensor.imag) * noise_std
        noise = torch.complex(noise_real, noise_imag)
        augmented = signal_tensor + noise
        
    elif augmentation_type == 'phase_shift':
        # Apply random phase shift
        phase_shift = kwargs.get('phase_shift', None)
        if phase_shift is None:
            phase_shift = torch.rand(1) * 2 * np.pi  # Random phase 0 to 2π
        
        # Convert phase_shift to tensor if it's a scalar
        if not isinstance(phase_shift, torch.Tensor):
            phase_shift = torch.tensor(phase_shift, dtype=torch.float32)
        
        # Create complex exponential
        phase_factor = torch.exp(1j * phase_shift.to(torch.complex64))
        augmented = signal_tensor * phase_factor
        
    elif augmentation_type == 'amplitude_scale':
        # Scale amplitude randomly
        scale_range = kwargs.get('scale_range', (0.8, 1.2))
        scale_factor = torch.rand(1) * (scale_range[1] - scale_range[0]) + scale_range[0]
        augmented = signal_tensor * scale_factor
        
    elif augmentation_type == 'frequency_shift':
        # Apply frequency shift (requires time axis information)
        freq_shift = kwargs.get('freq_shift', 0.1)  # Normalized frequency shift
        sample_rate = kwargs.get('sample_rate', 1.0)
        
        # Create time vector
        if signal_tensor.ndim == 1:
            t = torch.arange(len(signal_tensor), dtype=torch.float32) / sample_rate
        else:
            t = torch.arange(signal_tensor.shape[-1], dtype=torch.float32) / sample_rate
            t = t.unsqueeze(0).expand(signal_tensor.shape[:-1] + (-1,))
        
        # Apply frequency shift
        shift_factor = torch.exp(1j * 2 * np.pi * freq_shift * t.to(torch.complex64))
        augmented = signal_tensor * shift_factor
    
    return augmented


def apply_preprocessing_pipeline(rf_data: Union[np.ndarray, torch.Tensor],
                               extract_iq: bool = True,
                               compute_psd_features: bool = True,
                               normalization: str = 'zscore',
                               outlier_handling: str = 'clip',
                               augmentation: Optional[str] = None,
                               **kwargs) -> dict:
    """
    Apply complete preprocessing pipeline to RF data.
    
    Args:
        rf_data: Complex RF signal data
        extract_iq: Whether to extract I/Q channels
        compute_psd_features: Whether to compute PSD features
        normalization: Normalization method ('zscore', 'minmax', 'robust')
        outlier_handling: How to handle outliers ('clip', 'remove', 'none')
        augmentation: Optional augmentation to apply
        **kwargs: Additional parameters for specific operations
        
    Returns:
        dict: Dictionary containing processed features
        
    Raises:
        ValueError: If parameters are invalid
    """
    results = {}
    
    # Input validation
    validate_rf_input(rf_data)
    
    # Convert to torch tensor
    if isinstance(rf_data, np.ndarray):
        rf_tensor = torch.from_numpy(rf_data)
    else:
        rf_tensor = rf_data.clone()
    
    # Apply augmentation if specified
    if augmentation is not None:
        rf_tensor = augment_rf_signal(rf_tensor, augmentation, **kwargs)
        results['augmented_signal'] = rf_tensor
    
    # Extract I/Q channels
    if extract_iq:
        iq_channels = extract_iq_channels(rf_tensor)
        results['iq_channels'] = iq_channels
        
        # Apply normalization to I/Q channels
        if normalization == 'zscore':
            iq_normalized = normalize_zscore(iq_channels, axis=-1)
        elif normalization == 'minmax':
            iq_normalized = normalize_minmax(iq_channels, axis=-1)
        elif normalization == 'robust':
            iq_normalized = robust_normalize(iq_channels, axis=-1)
        else:
            iq_normalized = iq_channels
        
        results['iq_normalized'] = iq_normalized
    
    # Compute PSD features
    if compute_psd_features:
        psd_params = {k: v for k, v in kwargs.items() if k in ['nperseg', 'noverlap', 'nfft', 'fs']}
        psd_features = compute_psd(rf_tensor, **psd_params)
        results['psd_features'] = psd_features
        
        # Apply normalization to PSD features
        if normalization == 'zscore':
            psd_normalized = normalize_zscore(psd_features, axis=-1)
        elif normalization == 'minmax':
            psd_normalized = normalize_minmax(psd_features, axis=-1)
        elif normalization == 'robust':
            psd_normalized = robust_normalize(psd_features, axis=-1)
        else:
            psd_normalized = psd_features
        
        results['psd_normalized'] = psd_normalized
    
    # Handle outliers
    if outlier_handling != 'none' and 'iq_normalized' in results:
        outlier_mask = detect_outliers(results['iq_normalized'])
        results['outlier_mask'] = outlier_mask
        
        if outlier_handling == 'clip':
            # Clip outliers to reasonable bounds
            iq_clipped = torch.clamp(results['iq_normalized'], -3, 3)  # 3-sigma clipping
            results['iq_normalized'] = iq_clipped
        elif outlier_handling == 'remove':
            # Mark samples with outliers for removal
            sample_has_outliers = torch.any(outlier_mask.view(outlier_mask.shape[0], -1), dim=1)
            results['outlier_samples'] = sample_has_outliers
    
    return results