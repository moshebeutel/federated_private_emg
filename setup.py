"""Package configuration."""
from setuptools import find_packages, setup

setup(
    name="fed_prv_emg",
    version="0.0.3",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
