from setuptools import setup, find_packages

setup(
    name="ffenv",
    version="0.1",
    packages=find_packages(where='src'),
    install_requires=[
        "gymnasium",
        "numpy"
    ]
)
