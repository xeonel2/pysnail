#!/usr/bin/env python
#from distutils.core import setup
#from distutils.extension import Extension
from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [ 
        Extension(
            name = "snail._st",
            sources = ["src/pyst.pyx", "src/shiftpitch.cpp"],
            include_dirs = [ numpy.get_include(), "/usr/include/soundtouch", "/usr/local/include/soundtouch"],
            libraries = ["SoundTouch"],
            language = "c++",
            ),

        Extension(
            name = 'snail._yin',
            sources = ['src/pyin.pyx','src/Yin.c'],
            include_dirs = [numpy.get_include()],
            library_dirs = [],
            )
    ]

setup(
    version='1.0',
    description='Pitch-related functions.',
    author='Christopher A. Brown',
	author_email = "cbrown1@pitt.edu",
	maintainer = "Christopher Brown",
	maintainer_email = "cbrown1@pitt.edu",
	url = "http://pysnail.googlecode.com/",
    name = "snail",
    packages = ['snail'],
    cmdclass = {'build_ext':build_ext},
    ext_modules = ext_modules,
)
