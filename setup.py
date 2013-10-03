# -*- coding: utf-8 -*-

# Copyright (c) 2010-2013 Christopher Brown
#
# This file is part of Snail.
#
# Snail is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Snail is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Snail.  If not, see <http://www.gnu.org/licenses/>.
#
# Comments and/or additions are welcome. Send e-mail to: cbrown1@pitt.edu.
#

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
