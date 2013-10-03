/*
 Copyright (c) 2010-2013 Christopher Brown

 This file is part of Snail.

 Snail is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 Snail is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with Snail.  If not, see <http://www.gnu.org/licenses/>.

 Comments and/or additions are welcome. Send e-mail to: cbrown1@pitt.edu.

*/

#ifndef SHIFTPITCH_H
#define SHIFTPITCH_H
#include "SoundTouch.h"
using namespace soundtouch;
SAMPLETYPE *shiftPitch(
        SAMPLETYPE *sig_in, int sig_in_len, int nChannels, 
        int fs, float alpha, int quick, int aa, 
        int bufferSize, 
        int *sig_out_len
        );
double *
shiftPitch_double(
        double *sig_in, int sig_in_len, int nChannels, 
        int fs, float alpha, int quick, int aa, 
        int bufferSize, 
        int *sig_out_len); 
void shiftPitch_release(double *sig_out);
#endif
