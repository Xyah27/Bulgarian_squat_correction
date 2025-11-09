"""
Setup script for Bulgarian Split Squat Posture Analysis
"""

from setuptools import setup, find_packages
from pathlib import Path

# Leer README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Leer requirements
requirements = []
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="bulgarian-split-squat",
    version="1.0.0",
    author="Tu Nombre",
    author_email="tu.email@example.com",
    description="Sistema de análisis de postura para Bulgarian Split Squat usando BiGRU+Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tu-usuario/bulgarian-split-squat",
    project_urls={
        "Bug Tracker": "https://github.com/tu-usuario/bulgarian-split-squat/issues",
        "Documentation": "https://github.com/tu-usuario/bulgarian-split-squat/docs",
        "Source Code": "https://github.com/tu-usuario/bulgarian-split-squat",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinx-autodoc-typehints>=1.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bulgarian-squat-train=scripts.training.train_bigru:main",
            "bulgarian-squat-webcam=scripts.inference.run_webcam:main",
        ],
    },
    include_package_data=True,
    package_data={
        "bulgarian_squat": ["*.yaml", "*.json"],
    },
    keywords=[
        "pose estimation",
        "exercise analysis",
        "computer vision",
        "deep learning",
        "BiGRU",
        "attention mechanism",
        "mediapipe",
        "pytorch",
    ],
)
