#!/usr/bin/env python
#from distutils.core import setup
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
        Extension(
            name = 'snail.yin',
            sources = ['src/pyin.pyx','src/Yin.c'],
            include_dirs = [numpy.get_include()],
            libraries = [],
            library_dirs = [],
            language = "c",
            ),
        
        Extension(
            name = "snail.st",
            sources = ["src/pyst.pyx", "src/shiftpitch.cpp"],
            libraries = ["SoundTouch"],
            include_dirs = [ numpy.get_include(), '/usr/include/soundtouch', '/usr/local/include/soundtouch', 'soundtouch/include'],
            library_dirs = ['usr/local/lib/', '/usr/lib', 'soundtouch/lib'],
            language = "c++",
            )
        ]

setup(
    version='1.0',
    description='A few pitch-related functions.',
    author='Christopher A. Brown',
	author_email = "cbrown1@pitt.edu",
	maintainer = "Christopher Brown",
	maintainer_email = "cbrown1@pitt.edu",
	url = "http://pysnail.googlecode.com/",
    name = "snail",
    py_modules = ['snail.st', 'snail.yin'],
    cmdclass = {'build_ext':build_ext},
    ext_modules = ext_modules
)
