from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setup(
    name='factor_backtester',  
    version='1.0.0',  
    author='AlfredCYL',  
    author_email='alfred.yl@outlook.com', 
    description='A tool for factor backtesting',  
    long_description=long_description,  
    long_description_content_type='text/markdown',  
    url='https://github.com/AlfredCYL/factor_backtester',  
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
        'numba'
    ],
    include_package_data=False, # exclude the test data
)
