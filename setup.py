from setuptools import setup, find_packages

setup(
    name='pdf_extractor',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pillow',
        'unstructured[pdf]'
    ],
    entry_points={
        'console_scripts': [
            'extract-content=pdf_extractor:extract_whole_content',
            'extract-components=pdf_extractor:extract_by_components',
        ],
    },
)
