from setuptools import setup, find_packages

setup(
    name='pdf_extractor',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'PyMuPDF==1.23.26',
        'unstructured~=0.14.2',
        'unstructured[pdf]'
    ],
)
