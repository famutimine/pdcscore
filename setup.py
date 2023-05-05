from setuptools import setup, find_packages
import os

dir_path = r'C:\Users\famut\pdc_base_folder'
file_name = 'README.md'

file_path = os.path.join(dir_path, file_name)

with open(file_path, encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pdcscore',
    version='1.0.9',
    description='A package to facilitate efficient and accurate calculation of the medication adherence metric "Proportion of Days Covered" or "PDC".',
    long_description=long_description,
    long_description_content_type='text/markdown',
    # url='https://github.com/<username>/<repository>',
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
