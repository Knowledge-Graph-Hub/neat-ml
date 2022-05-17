import os
import re

from codecs import open as copen  # to use a consistent encoding
from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

# get the long description from the relevant file
with copen(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def read(*parts):
    with copen(os.path.join(here, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


__version__ = "0.0.1"

test_deps = [
    "pytest",
    "pytest-cov",
    "coveralls",
    "validate_version_code",
    "codacy-coverage",
    "parameterized",
    "mypy",
]

extras = {
    "test": test_deps,
}

setup(
    name="neat",
    version=__version__,
    description="Neural-network Embedding All the Things",
    long_description=long_description,
    url="https://github.com/Knowledge-Graph-Hub/NEAT",
    author="deepak.unni3@lbl.gov, justaddcoffee+github@gmail.com",
    author_email="Deepak Unni, Justin Reese",
    python_requires=">=3.7",
    license="BSD-3",
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Beta",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(exclude=["contrib", "docs", "tests*"]),
    entry_points={"console_scripts": ["neat=neat.cli:cli"]},
    tests_require=test_deps,
    install_requires=[
        "tqdm",
        "click",
        "pyyaml",
        "tensorflow",
        "ensmallen==0.7.0.dev7",
        "embiggen==0.10.0.dev3",
        "opencv-python",  # for embiggen's 4d tSNEs
        "numpy",
        "matplotlib",
        "sanitize_ml_labels",
        "sklearn",
        "pandas",
        "transformers",
        "torch",
        "boto3",
        "botocore",
        "validators",
        "linkml",
    ],
    extras_require=extras,
    https://github.com/Knowledge-Graph-Hub/NEAT_schema/archive/refs/heads/main.zip
)
