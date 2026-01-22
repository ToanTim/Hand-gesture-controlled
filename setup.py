"""
Setup script for hand-gesture-control package.

For modern installations, prefer using pyproject.toml:
    pip install -e .

This setup.py is provided for backward compatibility.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="hand-gesture-control",
    version="0.1.0",
    description="Hand gesture recognition and control system with MLOps best practices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/hand-gesture-control",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/hand-gesture-control/issues",
        "Documentation": "https://github.com/yourusername/hand-gesture-control/blob/main/README.md",
        "Source Code": "https://github.com/yourusername/hand-gesture-control",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "mediapipe>=0.8.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "isort>=5.10.0",
        ],
        "torch": [
            "torch>=1.10.0",
        ],
        "tensorflow": [
            "tensorflow>=2.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gesture-control=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="gesture-recognition computer-vision mlops hand-tracking",
    zip_safe=False,
    include_package_data=True,
)
