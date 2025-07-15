from setuptools import setup, find_packages

setup(
    name="ffenv",
    version="0.1",
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    py_modules=["candidates_utils", "fire_state", "firefighter", "tree_generator", "tree_utils"],
    install_requires=[
        "gymnasium",
        "numpy"
    ]
)
