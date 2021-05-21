from setuptools import setup, find_packages
import os
import sys


if sys.version_info[0] < 3:
    with open('README.md') as f:
        long_description = f.read()
else:
    with open('README.md', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='pytorch-attention',
    version='1.0',
    description='PyTorch Simple Attention',
    author='Javier Lorenzo Díaz',
    url='https://github.com/javierlorenzod/pytorch-attention-mechanism',
    license='Apache 2.0',
    long_description_content_type='text/markdown',
    long_description=long_description,
    packages=find_packages(), # automatically detect all subpackages and submodules, so long as they contain an __init__.py file
    install_requires=[
        'torch>=1.7.1'
    ]
)