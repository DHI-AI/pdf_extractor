from setuptools import setup, find_packages

setup(
    name='pdf_extractor',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'PyMuPDF',
        'unstructured',
        'unstructured[pdf]'
    ],
)
