#!/usr/bin/env python
import setuptools
from distutils.core import setup

setup(
    name='dagtasets',
    version='0.1.2dev',
    packages=['dagtasets','scenethecizer','lm_util'],
    scripts=['bin/scenethesize'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
    author='Anguelos Nicolaou',
    author_email='anguelos.nicolaou@gmail.com',
    url='https://github.com/anguelos/dagtasets',
    package_data={'scenethecizer': ["data/backgrounds/paper_texture.jpg","data/corpora/01_the_ugly_duckling.txt"]},
    install_requires=[
        'torch','torchvision','numpy','scikit-image','opencv-contrib-python','scipy'
    ],
)