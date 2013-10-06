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
