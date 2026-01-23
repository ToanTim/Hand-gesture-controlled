"""
Recognizers package containing gesture recognition models, utilities, and evaluation tools.
"""

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
