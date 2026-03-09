# Copyright (c) Meta Platforms, Inc. and affiliates.
from setuptools import find_packages, setup

setup(
    name="lst",
    version="0.1.0",
    description="Latent Speech-Text Transformer",
    author="Meta Platforms, Inc. and affiliates.",
    url="https://github.com/facebookresearch/lst",
    packages=find_packages(),
    install_requires=["sentencepiece", "tiktoken", "xformers"],
)
