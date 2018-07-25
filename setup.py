from distutils.core import setup

setup(
    name='spots',
    version='1.0',
    description='Google Location History utilities',
    author='Lucas Cardozo',
    author_email='lucasecardozo@gmail.com',
    packages=['spots'],
    install_requires=[
        'scikit-learn',
        'scipy',
        'pandas',
        'haversine'
    ]
)
