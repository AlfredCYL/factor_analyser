from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setup(
    name='factor_analyser',  
    version='1.0',  
    author='AlfredCYL',  
    author_email='alfred.yl@outlook.com', 
    description='A tool for factor development and analysis.',  
    long_description=long_description,  
    long_description_content_type='text/markdown',  
    url='https://github.com/AlfredCYL/factor_analyser',  
    packages=find_packages(), 
    classifiers=[  
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'statsmodels',
        'tabulate',
        'numba',
        'scipy',
        'seaborn',
        'scikit-learn',
        'plotly'
    ],
    include_package_data=False, # exclude the test data
)