from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="ntl-detection-system",
    version="1.0.0",
    author="Your Organization",
    author_email="info@your-org.com",
    description="Hybrid AI System for Non-Technical Loss Detection in Smart Grids",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ZuVn/DoAN_ChuyenNganh",
    packages=find_packages(exclude=["tests", "notebooks", "docs"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.4.1",
        ],
        "api": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.1",
        ],
        "gpu": [
            "tensorflow-gpu>=2.13.0",
        ],
        "all": [
            "pytest>=7.4.0",
            "fastapi>=0.100.0",
            "streamlit>=1.25.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ntl-detect=scripts.run_hybrid_detection:main",
            "ntl-train=scripts.train_ml_model:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.csv"],
    },
    zip_safe=False,
)