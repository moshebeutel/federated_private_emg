"""Package configuration."""
from setuptools import find_packages, setup

def requirements():
    list_requirements = []
    with open('/home/user1/GIT/pFedGP/requirements.txt') as f:
        for line in f:
            list_requirements.append(line.rstrip())
    return list_requirements

setup(
    name="federated_private_emg",
    version="0.0.13",
    packages=find_packages(where="federated_private_emg"),
    package_dir={"": "federated_private_emg"},
    install_requires=requirements()
)