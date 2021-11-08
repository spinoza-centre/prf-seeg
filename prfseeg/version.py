from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 0
_version_micro = 'dev1'  # use '' for first of series, number for 1 and above
_version_extra = False
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: Linux",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "prf-seeg - Intracranial EEG pRF experiment."
# Long description will go up on the pypi page
long_description = """
Structure loosely based on Shablona by Ariel Rokem: https://github.com/uwescience/shablona
"""

NAME = "prfseeg"
MAINTAINER = "Tomas Knapen"
MAINTAINER_EMAIL = "tknapen@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/spinoza-centre/prf-seeg"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Tomas Knapen"
AUTHOR_EMAIL = "tknapen@gmail.com"
PLATFORMS = "Linux"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DIR = {"": "prfseeg"}
PACKAGE_DATA = {'prfseeg': [pjoin('notebooks', '*')]}
REQUIRES = ["numpy", "scipy", "nibabel", "nilearn", "h5py", "pyyaml", "pandas", 'mne']
DEP_LINKS = []
PYTHON_REQUIRES = ">= 3.9"