"""
Hand Gesture Control - MLOps-ready gesture recognition system.
"""

__version__ = "0.1.0"
__author__ = "Expert Python Engineer & MLOps Specialist"

# Package level imports for convenience (lazy loading to avoid import errors)
def __getattr__(name):
    """Lazy import of submodules."""
    if name == "GestureClassificationMetrics":
        from recognizers.utils.metrics import GestureClassificationMetrics
        return GestureClassificationMetrics
    elif name == "load_hagrid_samples":
        from src.recognizers.utils.loaders import load_hagrid_samples
        return load_hagrid_samples
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "GestureClassificationMetrics",
    "load_hagrid_samples",
]
