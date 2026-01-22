"""
Performance Metrics Module
Provides reusable functions to measure and monitor model performance and system resource usage.

This module includes utilities for measuring:
- Inference latency (ms/frame)
- Frames per second (FPS)
- Model load time
- CPU and GPU utilization
- Memory usage
- Model size on disk and in memory
"""

import time
import psutil
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


class PerformanceMetrics:
    """Class to track and compute various performance metrics"""
    
    def __init__(self):
        """Initialize performance metrics tracker"""
        self.latencies = []
        self.timestamps = []
        self.memory_usage = []
        self.cpu_usage = []
        self.gpu_usage = []
        self.start_time = None
        self.process = psutil.Process()
        
    def reset(self):
        """Reset all metrics"""
        self.latencies.clear()
        self.timestamps.clear()
        self.memory_usage.clear()
        self.cpu_usage.clear()
        self.gpu_usage.clear()
        self.start_time = None


def measure_model_load_time(load_function, *args, **kwargs) -> Tuple[float, any]:
    """
    Measure the time taken to load a model
    
    Args:
        load_function: Function to call for loading the model
        *args: Arguments to pass to load_function
        **kwargs: Keyword arguments to pass to load_function
        
    Returns:
        Tuple of (load_time_ms, loaded_model)
    """
    start_time = time.perf_counter()
    model = load_function(*args, **kwargs)
    end_time = time.perf_counter()
    
    load_time_ms = (end_time - start_time) * 1000
    return load_time_ms, model


def measure_inference_latency(inference_function, *args, **kwargs) -> Tuple[float, any]:
    """
    Measure the latency of a single inference call
    
    Args:
        inference_function: Function to call for inference
        *args: Arguments to pass to inference_function
        **kwargs: Keyword arguments to pass to inference_function
        
    Returns:
        Tuple of (latency_ms, inference_result)
    """
    start_time = time.perf_counter()
    result = inference_function(*args, **kwargs)
    end_time = time.perf_counter()
    
    latency_ms = (end_time - start_time) * 1000
    return latency_ms, result


def calculate_fps(latencies: List[float]) -> float:
    """
    Calculate average FPS from a list of latencies
    
    Args:
        latencies: List of latency values in milliseconds
        
    Returns:
        Average FPS
    """
    if not latencies:
        return 0.0
    
    avg_latency_seconds = np.mean(latencies) / 1000
    if avg_latency_seconds == 0:
        return 0.0
    
    return 1.0 / avg_latency_seconds


def calculate_fps_over_time(timestamps: List[float], window_size: int = 30) -> List[float]:
    """
    Calculate FPS over a sliding window of frames
    
    Args:
        timestamps: List of timestamps when each frame was processed
        window_size: Number of frames to use for FPS calculation
        
    Returns:
        List of FPS values over time
    """
    if len(timestamps) < 2:
        return []
    
    fps_values = []
    for i in range(1, len(timestamps)):
        start_idx = max(0, i - window_size)
        window_timestamps = timestamps[start_idx:i+1]
        
        if len(window_timestamps) > 1:
            time_diff = window_timestamps[-1] - window_timestamps[0]
            if time_diff > 0:
                fps = (len(window_timestamps) - 1) / time_diff
                fps_values.append(fps)
            else:
                fps_values.append(0.0)
        else:
            fps_values.append(0.0)
    
    return fps_values


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics
    
    Returns:
        Dictionary with memory usage information in MB
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        'rss': memory_info.rss / (1024 * 1024),  # Resident Set Size in MB
        'vms': memory_info.vms / (1024 * 1024),  # Virtual Memory Size in MB
        'percent': process.memory_percent(),
        'available': psutil.virtual_memory().available / (1024 * 1024),  # Available RAM in MB
        'total': psutil.virtual_memory().total / (1024 * 1024)  # Total RAM in MB
    }


_process = psutil.Process() 
def get_cpu_usage(interval: Optional[float] = None) -> Dict[str, any]:
    """
    Get current CPU usage statistics including per-core data.

    Args:
        interval: Sampling interval in seconds. None means since the last call.
    """
    
    per_cpu_percent = psutil.cpu_percent(interval=interval, percpu=True)
    per_cpu_freq = psutil.cpu_freq(percpu=True) if psutil.cpu_freq(percpu=True) else []

    return {
        'process_percent': _process.cpu_percent(interval=interval),
        'system_percent': psutil.cpu_percent(interval=interval),
        'cpu_count': psutil.cpu_count(),
        'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0.0,
        'per_cpu_percent': per_cpu_percent,
        'per_cpu_freq': [freq.current for freq in per_cpu_freq] if per_cpu_freq else []
    }


def get_gpu_usage() -> Dict[str, any]:
    """
    Get current GPU usage statistics (if available)
    
    Returns:
        Dictionary with GPU usage information or None if no GPU available
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        
        gpu_info = {
            'available': True,
            'memory_used': mem_info.used / (1024 * 1024),  # MB
            'memory_total': mem_info.total / (1024 * 1024),  # MB
            'memory_percent': (mem_info.used / mem_info.total) * 100,
            'gpu_utilization': utilization.gpu,
            'memory_utilization': utilization.memory
        }
        
        pynvml.nvmlShutdown()
        return gpu_info
        
    except (ImportError, Exception) as e:
        return {
            'available': False,
            'error': str(e)
        }


def get_model_size(model_path: str) -> Dict[str, float]:
    """
    Get model file size
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dictionary with model size information in MB
    """
    if not os.path.exists(model_path):
        return {
            'size_mb': 0.0,
            'exists': False,
            'path': model_path
        }
    
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    
    return {
        'size_mb': size_mb,
        'size_bytes': size_bytes,
        'exists': True,
        'path': model_path
    }


def estimate_model_memory_size(model) -> float:
    """
    Estimate the memory size of a loaded model
    
    Args:
        model: Loaded model object
        
    Returns:
        Estimated size in MB
    """
    try:
        # Try to get size for PyTorch models
        if hasattr(model, 'parameters'):
            total_params = sum(p.numel() for p in model.parameters())
            # Assuming float32 (4 bytes per parameter)
            size_mb = (total_params * 4) / (1024 * 1024)
            return size_mb
    except:
        pass
    
    try:
        # Try to get size using sys.getsizeof
        size_mb = sys.getsizeof(model) / (1024 * 1024)
        return size_mb
    except:
        return 0.0


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """
    Compute statistical metrics for a list of values
    
    Args:
        values: List of numerical values
        
    Returns:
        Dictionary with statistical metrics
    """
    if not values:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'p50': 0.0,
            'p95': 0.0,
            'p99': 0.0
        }
    
    np_values = np.array(values)
    
    return {
        'mean': float(np.mean(np_values)),
        'median': float(np.median(np_values)),
        'std': float(np.std(np_values)),
        'min': float(np.min(np_values)),
        'max': float(np.max(np_values)),
        'p50': float(np.percentile(np_values, 50)),
        'p95': float(np.percentile(np_values, 95)),
        'p99': float(np.percentile(np_values, 99))
    }


def check_performance_thresholds(metrics: Dict[str, any], 
                                  max_latency_ms: float = 50.0,
                                  min_fps: float = 20.0,
                                  max_memory_mb: float = 1000.0,
                                  max_cpu_percent: float = 80.0) -> Dict[str, List[str]]:
    """
    Check if performance metrics exceed specified thresholds
    
    Args:
        metrics: Dictionary of computed metrics
        max_latency_ms: Maximum acceptable latency in milliseconds
        min_fps: Minimum acceptable FPS
        max_memory_mb: Maximum acceptable memory usage in MB
        max_cpu_percent: Maximum acceptable CPU usage percentage
        
    Returns:
        Dictionary with warnings for exceeded thresholds
    """
    warnings = []
    errors = []
    info = []
    
    # Check latency
    if 'latency' in metrics and 'mean' in metrics['latency']:
        mean_latency = metrics['latency']['mean']
        if mean_latency > max_latency_ms:
            warnings.append(f"Mean latency ({mean_latency:.2f}ms) exceeds threshold ({max_latency_ms}ms)")
        else:
            info.append(f"Latency is within acceptable range ({mean_latency:.2f}ms)")
    
    # Check FPS
    if 'fps' in metrics and 'average' in metrics['fps']:
        avg_fps = metrics['fps']['average']
        if avg_fps < min_fps:
            warnings.append(f"Average FPS ({avg_fps:.2f}) is below threshold ({min_fps})")
        else:
            info.append(f"FPS is within acceptable range ({avg_fps:.2f})")
    
    # Check memory
    if 'memory' in metrics and 'max' in metrics['memory']:
        max_memory = metrics['memory']['max']
        if max_memory > max_memory_mb:
            errors.append(f"Max memory usage ({max_memory:.2f}MB) exceeds threshold ({max_memory_mb}MB)")
        else:
            info.append(f"Memory usage is within acceptable range ({max_memory:.2f}MB)")
    
    # Check CPU
    if 'cpu' in metrics and 'max' in metrics['cpu']:
        max_cpu = metrics['cpu']['max']
        if max_cpu > max_cpu_percent:
            warnings.append(f"Max CPU usage ({max_cpu:.2f}%) exceeds threshold ({max_cpu_percent}%)")
        else:
            info.append(f"CPU usage is within acceptable range ({max_cpu:.2f}%)")
    
    return {
        'warnings': warnings,
        'errors': errors,
        'info': info
    }


class PerformanceTracker:
    """Context manager for tracking performance metrics during execution"""
    
    def __init__(self, name: str = "Task"):
        """
        Initialize performance tracker
        
        Args:
            name: Name of the task being tracked
        """
        self.name = name
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        
    def __enter__(self):
        """Start tracking"""
        self.start_time = time.perf_counter()
        self.start_memory = get_memory_usage()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop tracking and print results"""
        self.end_time = time.perf_counter()
        self.end_memory = get_memory_usage()
        
        duration_ms = (self.end_time - self.start_time) * 1000
        memory_delta = self.end_memory['rss'] - self.start_memory['rss']
        
        print(f"\n[{self.name}] Performance Summary:")
        print(f"  Duration: {duration_ms:.2f}ms")
        print(f"  Memory Delta: {memory_delta:+.2f}MB")
        print(f"  Final Memory: {self.end_memory['rss']:.2f}MB")
