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


def get_pitch(sig, double fs, double threshold = .15, int buffer_size = 2048, overlap = 0):
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
