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

# distutils: language = c++
# distutils: sources = Yin.cpp
#
#usage: get_pitch(array, sampleRate)
#
import numpy as np
cimport numpy as cnp
cnp.import_array()

cdef extern from "Yin.h":
    cdef struct _Yin:
        pass
    ctypedef _Yin Yin
    void Yin_init(Yin *yin, double sampleRate, int bufferSize, float threshold)
    double Yin_getPitch(Yin *yin, double *buffer)
    double Yin_getProbability(Yin *yin)
    void Yin_quit(Yin *yin)


def get_pitch(sig, double fs, double threshold, int buffer_size, overlap):
    """
    Estimates the pitch of a signal.

    Parameters
    ----------
    sig : array
        The input signal. Must be 1d.
    fs : int
        The sampling frequency.
    threshold : float
        Allowed uncertainty. Values can be 0 <= 1.
		(e.g 0.05 will return a pitch with ~95% probability).
    buffer_size : int
        The analysis buffer size, in samples. Default = 2048.

    Returns
    -------
    pitch : array
        The pitch track. Will be of size (sig.size/buffer_size).
    probability : array
        For each pitch estimate, the certainty of the accuracy. 
		Values will be 0 <= 1. Will be same size as pitch.

    Notes
    -----
	Uses the Yin algorithm, a well-established autocorrelation 
	based pitch algorith. Read a paper on Yin here:
	http://audition.ens.fr/adc/pdf/2002_JASA_YIN.pdf

	The C implementation of Yin was downloaded on 2013-09-21 from:
	https://github.com/ashokfernandez/Yin-Pitch-Tracking
	and has been modified significantly. It now works on doubles,
	and has no (known) memory leaks. There are a few other tweaks 
	as well.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim = 1, mode = 'c'] sig_in
    cdef Yin yin
    cdef int remains
    if overlap >= buffer_size:
        return None,None

    sig_in = np.ascontiguousarray(sig, dtype=np.float64)
    
    pitch_array = np.zeros(0, dtype = np.float64)
    prob_array = np.zeros(0, dtype = np.float64)

    pos = 0
    remains = sig_in.size - pos
    while remains > 0:
        if remains >= buffer_size:
            Yin_init(&yin, fs, buffer_size, threshold) 
            #Optimization: do not copy, move the pointer to the position
            pitch = Yin_getPitch(&yin, <double *>&(sig_in.data[8*pos]))
            prob = Yin_getProbability(&yin)
            pitch_array = np.append(pitch_array, pitch)
            prob_array = np.append(prob_array, prob)
            pos += buffer_size - overlap
            remains -= buffer_size - overlap
            #Release the internal buffer to remove the memory leak.
            Yin_quit(&yin)
        else:
            #If the last buffer is smaller
            Yin_init(&yin, fs, remains, threshold)
            #reuse the old buffer without creating new memory leak
            #pos -= buffer_size - remains
            pitch = Yin_getPitch(&yin, <double *>&(sig_in.data[8*pos]))
            prob = Yin_getProbability(&yin)
            pitch_array = np.append(pitch_array, pitch)
            prob_array = np.append(prob_array, prob)
            #Release the internal buffer to remove the memory leak.
            Yin_quit(&yin)
            break
    return pitch_array, prob_array
