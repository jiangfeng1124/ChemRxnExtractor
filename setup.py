import os
from setuptools import setup, find_packages

__version__ = None

src_dir = os.path.abspath(os.path.dirname(__file__))
version_file = os.path.join(src_dir, 'chemrxnextractor', '_version.py')

with open(version_file, encoding='utf-8') as fd:
    exec(fd.read())

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='chemrxnextractor',
    author="Jiang Guo, Santiago Ibanez, Hanyu Gao",
    author_email="jiang_guo@csail.mit.edu",
    description='Chemical Reaction Extraction from Literature',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='Pre-trainin://github.com/jiangfeng1124/ChemRxnExtractor',
    version=__version__,
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'torch>=1.5.0',
        'tqdm>=4.36.0',
        'transformers>=3.0.2',
        'seqeval',
        'numpy>=1.18.0'
    ],
    keywords=[
        'chemistry',
        'information extraction',
        'reaction extraction',
        'natural language processing',
        'pre-training'
    ]
)
