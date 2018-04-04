#!/usr/bin/env python

from distutils.core import setup

setup(
    name='dagtasets',
    version='0.1dev',
    packages=['dagtasets',],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
    author='Anguelos Nicolaou',
    author_email='anguelos.nicolaou@gmail.com',
    url='https://github.com/anguelos/dagtasets',
    install_requires=[
        'torch','torchvision','numpy'
    ],
)

