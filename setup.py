from setuptools import setup, find_packages

setup(
    name="mirai",
    version="0.1",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
)
