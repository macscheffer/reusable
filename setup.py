from pathlib import Path
from setuptools import setup, find_packages

README = (Path(__file__).parent / "README.md").read_text()

setup(
    name='macs',
    version='0.1',
    description="Personal library of Python tools",
    long_description=README,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    author="mac scheffer",
    install_requires=["sklearn”, “pandas”]
)