// Copyright (c) 2010-2013 Christopher Brown
//
// This file is part of Snail.
//
// Snail is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Snail is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Snail.  If not, see <http://www.gnu.org/licenses/>.
//
// Comments and/or additions are welcome. Send e-mail to: cbrown1@pitt.edu.

extern "C" {
    #include <Python.h>
    #include <numpy/arrayobject.h>
    #include <stdio.h>
	#include "Yin.h"
}
#include <math.h>
#include <soundtouch/SoundTouch.h>
#include "snail.h"

static PyMethodDef _snailMethods[] = {
    {"shift_pitch", shift_pitch,  METH_VARARGS},
    {"get_pitch", get_pitch,  METH_VARARGS},
    {"get_st_LibVersion", get_st_LibVersion,  METH_VARARGS},
    {NULL, NULL}
};

extern "C" void init_snail ()
{
    (void) Py_InitModule("_snail", _snailMethods);
    import_array();
}

PyObject* get_st_LibVersion(PyObject* /*self*/, PyObject* /*args*/) {
    soundtouch::SoundTouch st = soundtouch::SoundTouch();
    return PyString_FromString(st.getVersionString());
}

static PyObject *
get_pitch (PyObject *self, PyObject *args)
{
    // Signal, pitch, probability arrays
    PyArrayObject *x, *y, *p;

    // Sample rate of `x`
    int fs;

    // Window buffer size in samples
    int buff_size_samples;

    // Allowed uncertainty.
    float threshold;

    // Pointers to the actual array data for processing
    int16_t *xarr;
    double *yarr;
    double *parr;

    // Processing buffer
    int16_t *buf_in;

    // Size, in samples, of the signal 
    int arr_size;

    // Counters
    int i, j, m;

	// _snail.get_pitch(sigi,int(fs),pitch,prob,buffer_size,threshold)
    // http://docs.python.org/c-api/arg.html
    if (!PyArg_ParseTuple(args, "O!iO!O!if",
                          &PyArray_Type, &x,
                          &fs,
                          &PyArray_Type, &y,
                          &PyArray_Type, &p,
                          &buff_size_samples,
                          &threshold)) {
        return NULL;
    }

    xarr = (int16_t *) x->data;
    yarr = (double *) y->data;
    parr = (double *) p->data;

    arr_size = PyArray_Size((PyObject*)x);

	buf_in = (int16_t *)malloc(buff_size_samples * sizeof(int16_t));

	Yin yin;
	Yin_init(&yin, buff_size_samples, threshold);
    m = 0;
    for (i = 0; i < arr_size; i += buff_size_samples)
    {
        for (j = 0; j < buff_size_samples; j++)
        {
            buf_in[j] = xarr[i+j];
        }
		yarr[m] = (double) Yin_getPitch(&yin, buf_in, fs);
		parr[m++] = (double) Yin_getProbability(&yin);
    }
    free(buf_in);
    Py_INCREF(Py_None);

    return Py_None;
}

static PyObject *
shift_pitch (PyObject *self, PyObject *args)
{
    // Input, output signals
    PyArrayObject *x, *y;

    // Count of received samples 
    int nSamples;

    // Sample rate of `x`
    int fs;

    // # channels in `x`
    int channels;

    // Window buffer size in samples
    int buff_size_samples;

    int buff_size_frames;

    // Amount of compression. alpha < 1
    float alpha;
	int quick;

    // Pointers to the actual array data for processing
    double *xarr;
    double *yarr;

    // Processing buffers
    float *buf_in;
    float *buf_out;

    // Size, in samples, of the signal
    int arr_size;

    // Counters
    int i, j, m;

	// _snail.shift_pitch(sig2, y, int(fs), ch, float(alpha), buffer_size, quick)
    // http://docs.python.org/c-api/arg.html
    if (!PyArg_ParseTuple(args, "O!O!iifii",
                          &PyArray_Type, &x,
                          &PyArray_Type, &y,
                          &fs,
                          &channels,
                          &alpha,
                          &buff_size_samples,
                          &quick)) {
        return NULL;
    }

    xarr = (double *) x->data;
    yarr = (double *) y->data;

    arr_size = PyArray_Size((PyObject*)x);

    buff_size_frames = buff_size_samples / channels;

    buf_in  = (float *) malloc(sizeof(float) * buff_size_samples);
    buf_out = (float *) malloc(sizeof(float) * buff_size_samples);

    soundtouch::SoundTouch st = soundtouch::SoundTouch();
    st.setSampleRate(fs);
    st.setChannels(channels);
	st.setSetting(SETTING_USE_QUICKSEEK, quick);

    st.setPitchOctaves(alpha);

    m = 0;
    for (i = 0; i < arr_size; i += buff_size_samples)
    {
        for (j = 0; j < buff_size_samples; j++)
        {
            buf_in[j] = (float) xarr[i+j];
        }
        st.putSamples(buf_in, buff_size_frames);
        do
        {
            nSamples = st.receiveSamples(buf_out, buff_size_frames);
            for (j = 0; j < nSamples * channels; j++)
            {
                yarr[m++] = (double) buf_out[j];
            }
        } while (nSamples != 0);
    }
    st.flush();
    do
    {
        nSamples = st.receiveSamples(buf_out, buff_size_frames);
        for (j = 0; j < nSamples * channels; j++)
        {
            yarr[m++] = (double) buf_out[j];
        }
    } while (nSamples != 0);

    st.clear();
    free(buf_in);
    free(buf_out);

    Py_INCREF(Py_None);

    return Py_None;
}