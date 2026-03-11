#!/usr/bin/env python3
from setuptools import setup, find_packages

# 读取版本号
version = "0.1.0"

setup(
    name="sam3",
    version=version,
    description="SAM3 (Segment Anything Model 3) implementation",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/sam3",
    author="Meta AI Research",
    license="MIT",
    python_requires=">=3.8",
    packages=find_packages(include=["sam3*"]),
    package_data={"sam3": ["assets/*.txt.gz"]},
    install_requires=[
        "timm>=1.0.17",
        "numpy>=1.26,<2",
        "tqdm",
        "ftfy==6.1.1",
        "regex",
        "iopath>=0.1.10",
        "typing_extensions",
        "huggingface_hub",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
