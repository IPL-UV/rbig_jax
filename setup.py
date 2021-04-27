#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys
from shutil import rmtree

from setuptools import Command, find_packages, setup


def parse_requirements_file(filename):
    with open(filename, encoding="utf-8") as fid:
        requires = [l.strip() for l in fid.readlines() if l]
    return requires


# Package meta-data.
NAME = "rbig_jax"
DESCRIPTION = "Gaussianization with JAX."
URL = "https://github.com/ipl-uv/rbig_jax"
EMAIL = "jemanjohnson34@gmail.com"
AUTHOR = "J. Emmanuel Johnson"
REQUIRES_PYTHON = ">=3.7.0"
VERSION = "0.1.0"

# What packages are required for this module to be executed?
REQUIRED = [
    # Standard Libraries
    "numpy>=1.19", "scipy>=1.5", "scikit-learn>=0.23",
    # JAX Libraries
    "jax>=0.2.11",
    "jaxlib>=0.1.62",
    "numpyro",
    "chex",
    "optax",
    "distrax>=0.0.1",
    "tfp-nightly",
    "tqdm",
    "einops",
    "tensor_annotations",
    "scikit-image",
    # PyTorch and Co
    "torch>=1.6",
    "torchvision>=0.9",
    "torchaudio>=0.8",
    "torchmetrics>=0.2.0",
    "pytorch-lightning>=1.2",
    "lightning-bolts>=0.3",
    # Testing & Formatting
    "black",
    "flake8",
    "isort>=5.0",
    "pytest>4.1",
    # Jupyter Notebooks
    "ipykernel",
    "nb_black",
    "pyprojroot",
    # Logging
    "loguru",
    "wandb,"
    # Plot Libraries
    "matplotlib>=3.3",
    "seaborn",
    "celluloid",
    "corner",
]

# What packages are optional?
EXTRAS = {
    "dev": ["black", "isort", "mypy"],
    "tests": ["pytest", "scikit-learn"],
    "extras": ["tqdm", "wandb", "scipy", "corner", "celluloid",],
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],
    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
#     setup_requires=["setuptools-yaml"],
#     metadata_yaml="environment.yml",
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    # $ setup.py publish support.
    cmdclass={"upload": UploadCommand},
)
