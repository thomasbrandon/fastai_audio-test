#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

from glob import glob
from os.path import basename
from os.path import splitext
from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="fastai_audio",
    version="0.0.1",
    author="Thomas Brandon",
    description="Fastai audio handling (test)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thomasbrandon/fastai_audio-test",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.6',
    install_requires=[ 'fastai>=1.0', 'librosa' ],
    tests_requires = [ 'pytest', 'pytest-mock', 'soundfile' ],
    classifiers=[
        'Development Status :: 1 - Planning',
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
)