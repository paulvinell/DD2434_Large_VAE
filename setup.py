# This script is for gcloud setup

from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='large_vae_experiment',
    version='0.1',
    install_requires=required,
    packages=find_packages(),
    include_package_data=True,
    description='Large VAE project'
)
