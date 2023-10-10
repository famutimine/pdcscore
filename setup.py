from setuptools import setup, find_packages
import os

from urllib.request import urlopen

with urlopen("https://raw.githubusercontent.com/famutimine/pdcscore/master/README.md") as fh:
    long_description = fh.read().decode()

setup(
    name='pdcscore',
    version='1.1.9',
    description='A package to facilitate efficient and accurate calculation of the medication adherence metric "Proportion of Days Covered" or "PDC".',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/famutimine/pdcscore',
    author='Daniel Famutimi MD, MPH',
    author_email='danielfamutimi@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: System Administrators',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    keywords='pdc calculator medication adherence',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
    ],
)
