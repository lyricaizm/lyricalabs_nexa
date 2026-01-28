from setuptools import setup, find_packages
import os

# README dosyasını oku
def read_readme():
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "XanaxWay API Python, Nexa LLM"

long_description = read_readme()

setup(
    name="xanaxway",
    version="0.1.4",
    packages=find_packages(),
    install_requires=["requests"],
    python_requires=">=3.8",
    description="XanaxWay API Python, Nexa LLM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://lyricalabs.com",
    author="XanaxWay",
    author_email="xanaxway@gmail.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
