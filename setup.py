"""
setup.py
========
Optional: makes the project pip-installable as a local package.

Usage:
    pip install -e .          # editable install (development)
    pip install .             # standard install
"""

from setuptools import setup, find_packages
from pathlib import Path

HERE = Path(__file__).parent
LONG_DESC = (HERE / "README.md").read_text(encoding="utf-8")

setup(
    name="emotion-recognition-speech",
    version="1.0.0",
    description="Deep learning pipeline for emotion recognition from speech (CNN, LSTM, CNN-LSTM, Transformer)",
    long_description=LONG_DESC,
    long_description_content_type="text/markdown",
    author="Arasu Kumar S",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    install_requires=[
        "tensorflow>=2.13.0",
        "librosa>=0.10.1",
        "soundfile>=0.12.1",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "streamlit>=1.28.0",
    ],
    extras_require={
        "mic": ["pyaudio>=0.2.13"],
        "app": ["streamlit-audiorec>=0.1.3"],
        "dev": ["ipykernel>=6.0.0", "notebook>=7.0.0", "pytest>=7.0"],
    },
    entry_points={
        "console_scripts": [
            "emotion-train=src.train:main",
            "emotion-predict=src.predict:main",
            "emotion-eval=src.evaluate:main",
            "emotion-app=app.streamlit_app:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
