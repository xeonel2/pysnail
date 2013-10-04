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
import os
from setuptools import setup, Extension
from distutils.sysconfig import get_python_lib
from Cython.Distutils import build_ext
import numpy
import platform

snail_data_files = []
snail_data_files_path = os.path.join(get_python_lib(), 'snail')

st_sources = ["src/pyst.pyx", "src/shiftpitch.cpp"]
st_libraries = ['SoundTouch']
st_library_dirs = ['soundtouch/lib']
st_include_dirs = [ numpy.get_include() ]

yin_sources = ['src/pyin.pyx','src/Yin.c']
yin_library_dirs = []
yin_include_dirs = [numpy.get_include()]

if platform.system() == "Windows":
    snail_data_files.append('soundtouch/dll/SoundTouch.dll')
    st_include_dirs.append('soundtouch/include')
else:
    st_include_dirs.append("/usr/include/soundtouch")
    st_include_dirs.append("/usr/local/include/soundtouch")

ext_modules = [ 
        Extension(
            name = "snail._st",
            sources = st_sources,
            include_dirs = st_include_dirs,
            library_dirs = st_library_dirs,
            libraries = st_libraries,
            language = "c++",
            ),

        Extension(
            name = 'snail._yin',
            sources = yin_sources,
            include_dirs = yin_include_dirs,
            library_dirs = yin_library_dirs,
            language = "c",
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
    data_files=[(snail_data_files_path, snail_data_files)],
    cmdclass = {'build_ext':build_ext},
    ext_modules = ext_modules,
)
