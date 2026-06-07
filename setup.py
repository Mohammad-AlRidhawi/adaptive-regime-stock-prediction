from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="adaptive-regime-stock-prediction",
    version="1.0.0",
    description="Adaptive regime-aware stock price prediction with autoencoder-gated dual node transformers and SAC control",
    author="Mohammad Al Ridhawi",
    author_email="malri039@uottawa.ca",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.10",
)
