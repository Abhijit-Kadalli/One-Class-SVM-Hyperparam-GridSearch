import os
from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), "README.md")) as readme:
    README = readme.read()

setup(
    name="gridsearch_ocsvm",
    version="0.0.1",
    author="Abhijit Kadalli",
    author_email="abhijitkadalli14@gmail.com",
    description="Grid Search for One-Class SVM",
    long_description="A package for grid search with One-Class SVM",
    long_description_content_type="text/markdown",
    url="https://github.com/Abhijit-Kadalli/One-Class-SVM-Hyperparam-GridSearch/tree/main/pythonPackage",
    packages=find_packages(),
    install_requires=[
        "joblib>=1.1.0",
        "matplotlib>=3.5.2",
        "numpy>=1.23.0",
        "scikit_learn>=1.3.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
