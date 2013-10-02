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
#include "pyin.h"

static PyMethodDef _pyinMethods[] = {
    {"get_pitch", get_pitch,  METH_VARARGS},
    {NULL, NULL}
};

extern "C" void init_yin ()
{
    (void) Py_InitModule("_yin", _pyinMethods);
    import_array();
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

