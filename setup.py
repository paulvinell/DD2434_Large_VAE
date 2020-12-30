# This script is for gcloud setup

from setuptools import find_packages
from setuptools import setup

required = ["keras", "sklearn", "numpy"]

setup(
    name='large_vae_experiment',
    version='0.1',
    install_requires=required,
    packages=find_packages(),
    include_package_data=True,
    description='Large VAE project'
)
