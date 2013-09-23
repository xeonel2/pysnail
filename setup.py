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
from distutils.core import setup, Extension
from distutils.sysconfig import get_python_inc, get_python_lib
import numpy
import os
import sys
sys.path.append('src')

inc_d = ['.',
         get_python_inc(),
         numpy.get_include()]

if os.name == 'posix':
    inc_d.append('/usr/local/include')

_snail = Extension('snail._snail',
                     include_dirs = inc_d,
                     libraries = ['SoundTouch'],
                     library_dirs = [get_python_lib(), 'soundtouch'],
                     sources = ['src/snail.cpp','src/Yin.c'])
setup (name = 'snail',
    version='1.0',
    description='An extremely light wrapper for soundtouch',
    author='Christopher A. Brown, Joseph K. Ranweiler',
	author_email = "cbrown1@pitt.edu",
	maintainer = "Christopher Brown",
	maintainer_email = "cbrown1@pitt.edu",
	url = "http://code.google.com/p/pysoundtouch/",
    packages = ['snail'],
    ext_modules = [_snail],
    )
