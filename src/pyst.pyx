# distutils: language = c++
# distutils: sources = shiftpitch.cpp
# distutils: libraries = SoundTouch
# distutils: include_dirs = /usr/include/soundtouch
#
#usage: shift_pitch(array, sampleRate, 1.0)
#
import numpy as np
cimport numpy as cnp
cnp.import_array()

cdef extern from "shiftpitch.h":
    double *shiftPitch_double(double *sig_in, int sig_in_len, int nChannels, 
        int fs, float alpha, int quick, int aa, 
        int bufferSize, 
        int *sig_out_len)

    void shiftPitch_release(double *sig_out)


def shift_pitch(sig, fs, alpha, quick=True, aa=True, buffer_size=2048):
    """
    Applies a specified amount of freq compression/expansion to a signal.

    Parameters
    ----------
    sig : array
        The input signal. Must be 1d.
    fs : int
        The sampling frequency.
    alpha : float
        The pitch change, in octaves. -1 = 1 octave of compression, 0 = no 
        change, 1 = 1 octave of expansion.
    quick : bool
        True = use soundtouch's 'quick' setting, which speeds up processing
        significantly, with a small degredation in quality. Default = True.
    aa : bool
        Whether to use soundtouch's anti-aliasing filter. Default = True.
    buffer_size : int
        The analysis buffer size, in samples. Default = 2048.

    Returns
    -------
    sig : array
        The signal with specified amount of freq compression/expansion applied.

    Notes
    -----
    Due to soundtouch's reliance on interpolation, the output buffer may 
    differ in length slightly from the input buffer.

    Works with mono or stereo data (1 or 2 rows).

    Uses the soundtouch library: http://www.surina.net/soundtouch/
    """
    cdef int sig_out_len
    cdef int nChannels
    cdef int c_quick
    cdef int c_aa
    if sig.ndim in [1, 2]:
        nChannels = sig.ndim
    else:
        return None
    if quick:
        c_quick = 1
    else:
        c_quick = 0
    if aa:
        c_aa = 1
    else:
        c_aa = 0

    #python -> c
    sig_in = np.ascontiguousarray(sig, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim = 1, mode = 'c'] sig_in_1d 
    if nChannels == 1:
        sig_in_1d = sig_in
    else:
        sig_in_1d = sig_in.reshape(sig_in.size)

    #call soundtouch and receive the c++ buffer
    cdef double * sig_out = shiftPitch_double(<double *>sig_in_1d.data, sig.shape[0], nChannels, fs, alpha, c_quick, c_aa, buffer_size, &sig_out_len)
    if sig_out == NULL:
        return None
    if nChannels == 1:
        out_shape = (sig_out_len,)
    else:
        out_shape = (sig_out_len, nChannels)
   
    #c -> python
    cdef nd = nChannels
    cdef cnp.npy_intp dims[2]
    dims[0] = out_shape[0]
    if nChannels == 2: dims[1] = out_shape[1]

    #Copy the result without bothering the user
    out = np.copy(cnp.PyArray_SimpleNewFromData(nd, dims, cnp.NPY_FLOAT64, <void *>sig_out))

    #release the c++ buffer
    shiftPitch_release(sig_out)
    return out
