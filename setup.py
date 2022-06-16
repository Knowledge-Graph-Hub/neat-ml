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


__version__ = "0.1.0"

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
    name="neat_ml",
    version=__version__,
    description="Neural-network Embedding All the Things",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/Knowledge-Graph-Hub/neat_ml",
    author="Deepak Unni, Justin Reese, J. Harry Caufield, Harshad Hegde",
    author_email="deepak.unni3@lbl.gov, justaddcoffee+github@gmail.com, jhc@lbl.gov, hhegde@lbl.gov",
    python_requires=">=3.8",
    license="BSD-3",
    include_package_data=True,
    classifiers=[],
    packages=find_packages(exclude=["contrib", "docs", "tests*"]),
    entry_points={"console_scripts": ["neat=neat_ml.cli:cli"]},
    tests_require=test_deps,
    install_requires=[
        # tensorflow and torch may also be required,
        # but should be installed separately
        # to avoid version conflicts
        "tqdm",
        "click",
        "pyyaml",
        "sklearn",
        "grape",
        "opencv-python",  # for embiggen's 4d tSNEs
        "numpy",
        "pandas",
        "transformers",
        "boto3",
        "botocore",
        "validators",
        "linkml",
        "neat-ml-schema",
        "linkml-validator"
    ],
    extras_require=extras,
)
