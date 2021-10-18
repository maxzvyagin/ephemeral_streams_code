"""Setup for the spaceray package."""

import setuptools


# with open('README.md') as f:
#     README = f.read()

setuptools.setup(
    author="Max Zvyagin",
    author_email="max.zvyagin7@gmail.com",
    name='ephemeral_streams',
    license="MIT",
    description='Ephemeral streams processing code',
    version='v0.2.6',
    long_description='https://github.com/maxzvyagin/ephemeral_streams_code',
    url='https://github.com/maxzvyagin/ephemeral_streams_code',
    packages=setuptools.find_packages(),
    python_requires=">=3.5",
    install_requires=["wandb", "torch", "pytorch-lightning", "segmentation_models_pytorch"]
)