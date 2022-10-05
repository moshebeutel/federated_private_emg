"""Package configuration."""
from setuptools import find_packages, setup

setup(
    name="federated_private_emg",
    version="0.0.7",
    packages=find_packages(where="federated_private_emg"),
    package_dir={"": "federated_private_emg"},
)