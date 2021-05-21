import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='pytorch-attention',
    version='1.0.0-rc-1',
    description='Simple many-to-one attention in PyTorch',
    author='Javier Lorenzo Díaz',
    url='https://github.com/javierlorenzod/pytorch-attention-mechanism',
    project_urls ={
        "Bug Tracker": "https://github.com/javierlorenzod/pytorch-attention-mechanism/issues",
    },
    classifiers =[
        "Programming Language :: Python :: 3"
        "Programming Language :: Python :: 3.8"
        "License :: OSI Approved :: Apache Software License"
        "Operating System :: OS Independent"
    ],
    long_description_content_type='text/markdown',
    long_description=long_description,
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8.5",
    install_requires=[
        'torch>=1.7.1'
    ],
)
