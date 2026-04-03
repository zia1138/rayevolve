"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series
"""
import numpy as np


def adaptive_filter(x, window_size=20):
    """
    Adaptive signal processing algorithm using sliding window approach.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (W samples)

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    output_length = len(x) - window_size + 1
    y = np.zeros(output_length)

    for i in range(output_length):
        window = x[i : i + window_size]
        y[i] = np.mean(window)

    return y


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Enhanced version with trend preservation using weighted moving average.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window

    Returns:
        y: Filtered output signal
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    output_length = len(x) - window_size + 1
    y = np.zeros(output_length)

    weights = np.exp(np.linspace(-2, 0, window_size))
    weights = weights / np.sum(weights)

    for i in range(output_length):
        window = x[i : i + window_size]
        y[i] = np.sum(window * weights)

    return y


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    if algorithm_type == "enhanced":
        return enhanced_filter_with_trend_preservation(input_signal, window_size)
    else:
        return adaptive_filter(input_signal, window_size)


def run_signal_processing(noisy_signal=None, signal_length=1000, noise_level=0.3, window_size=20):
    """
    Run the signal processing algorithm on a test signal.

    Args:
        noisy_signal: Input signal to filter (if provided, use this; otherwise generate)
        signal_length: Length if generating signal (for backward compatibility)
        noise_level: Noise level if generating signal (for backward compatibility)
        window_size: Window size for processing

    Returns:
        Dictionary containing results and metrics
    """
    if noisy_signal is not None:
        filtered_signal = process_signal(noisy_signal, window_size, "enhanced")
        return {
            "filtered_signal": filtered_signal,
            "clean_signal": None,
            "noisy_signal": None,
            "correlation": 0,
            "noise_reduction": 0,
            "signal_length": len(filtered_signal),
        }
    else:
        np.random.seed(42)
        t = np.linspace(0, 10, signal_length)
        clean_signal = (
            2 * np.sin(2 * np.pi * 0.5 * t)
            + 1.5 * np.sin(2 * np.pi * 2 * t)
            + 0.5 * np.sin(2 * np.pi * 5 * t)
            + 0.8 * np.exp(-t / 5) * np.sin(2 * np.pi * 1.5 * t)
        )
        trend = 0.1 * t * np.sin(0.2 * t)
        clean_signal += trend
        random_walk = np.cumsum(np.random.randn(signal_length) * 0.05)
        clean_signal += random_walk
        noise = np.random.normal(0, noise_level, signal_length)
        noisy_signal = clean_signal + noise

        filtered_signal = process_signal(noisy_signal, window_size, "enhanced")

        delay = window_size - 1
        aligned_clean = clean_signal[delay:]
        aligned_noisy = noisy_signal[delay:]
        min_length = min(len(filtered_signal), len(aligned_clean))
        filtered_signal = filtered_signal[:min_length]
        aligned_clean = aligned_clean[:min_length]
        aligned_noisy = aligned_noisy[:min_length]

        correlation = np.corrcoef(filtered_signal, aligned_clean)[0, 1] if min_length > 1 else 0
        noise_before = np.var(aligned_noisy - aligned_clean)
        noise_after = np.var(filtered_signal - aligned_clean)
        noise_reduction = (noise_before - noise_after) / noise_before if noise_before > 0 else 0

        return {
            "filtered_signal": filtered_signal,
            "clean_signal": aligned_clean,
            "noisy_signal": aligned_noisy,
            "correlation": correlation,
            "noise_reduction": noise_reduction,
            "signal_length": min_length,
        }
