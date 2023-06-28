import os

from setuptools import find_packages
from setuptools import setup

dir_path = os.path.dirname(os.path.realpath(__file__))


def read_requirements_file(filename):
    req_file_path = '%s/%s' % (dir_path, filename)
    with open(req_file_path) as f:
        return [line.strip() for line in f]

setup(
    name='robobuf',
    version='0.0.1',
    author='Mohan Kumar Srirama, Sudeep Dasari',
    license='MIT',
    packages=find_packages(),
    install_requires=read_requirements_file('requirements.txt')
)