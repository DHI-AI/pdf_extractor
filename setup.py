from setuptools import setup, find_packages

# Function to parse requirements.txt
def parse_requirements(filename):
    with open(filename, "r") as file:
        return file.read().splitlines()

setup(
    name='pdf_extractor',
    version='0.1.0',
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
)
