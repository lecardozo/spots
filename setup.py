from setuptools import setup
from os import path
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='spots',
    version='1.0.3',
    description='Google Location History utilities',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Lucas Cardozo',
    author_email='lucasecardozo@gmail.com',
    packages=['spots'],
    install_requires=[
        'scikit-learn',
        'pandas',
        'numba'
    ]
)
