# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Inspired from https://github.com/kennethreitz/setup.py

from pathlib import Path

from setuptools import setup, find_packages

NAME = 'flashy'
DESCRIPTION = 'Minimal solver for deep learning'

URL = 'https://github.com/fairinternal/flashy'
EMAIL = 'defossez@fb.com'
AUTHOR = 'Alexandre Défossez'
REQUIRES_PYTHON = '>=3.8.0'

for line in open('flashy/__init__.py'):
    line = line.strip()
    if '__version__' in line:
        context = {}
        exec(line, context)
        VERSION = context['__version__']

HERE = Path(__file__).parent

try:
    with open(HERE / "README.md", encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


REQUIRED = [i.strip() for i in open(HERE / 'requirements.txt') if not i.startswith('#')]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    package_data={"flashy": ["py.typed"]},
    install_requires=REQUIRED,
    include_package_data=True,
    extras_require={
        'dev': ['coverage', 'flake8', 'mypy', 'pdoc3', 'pytest', 'hydra_core'],
    },
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
