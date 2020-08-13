#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import sys

from setuptools import setup

version = "0.0.1"

with io.open("README.md", "r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

if sys.argv[-1] == "readme":
    print(readme)
    sys.exit()

setup(
    name="ganfluencer",
    version=version,
    description=(
        "Experimenting with GANs to generate 'beauty influencer' YouTube "
        "thumbnails"
    ),
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Emma Kwiecinska",
    author_email="ekwska0@protonmail.com",
    url="https://github.com/ekwska/ganfluencer",
    packages=["ganfluencer",],
    package_dir={"ganfluencer": "ganfluencer"},
    include_package_data=True,
    python_requires=">=3.7",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Topic :: Research",
    ],
    keywords=("GAN, Python, personal projects, DCGAN"),
)
