#include "shiftpitch.h"
#include <stdio.h>
/* extern "C" */
SAMPLETYPE *
shiftPitch(
        SAMPLETYPE *sig_in, int sig_in_len, int nChannels, 
        int fs, float alpha, int quick, int aa, 
        int bufferSize, 
        int *sig_out_len
        )//export to C lib
{
    if(!(sig_in&&sig_out_len)) return NULL;
    SoundTouch st;

    //setup the parameters
    st.setChannels(nChannels);
    
    st.setSampleRate(fs);
    st.setPitchOctaves(alpha);
    st.setSetting(SETTING_USE_QUICKSEEK, quick);
    st.setSetting(SETTING_USE_AA_FILTER, aa);

    SAMPLETYPE *sig_out = NULL;
    int sig_out_size = 0; /* num of SIMPLETYPE, not channels block */
    int remains;
    int num;

    //process
    int sig_in_pos = 0;/* index of the currenct channels block */
    int sig_out_pos = 0;
    int buffSizeSamples = bufferSize / nChannels;

    sig_out_size = sig_in_len;
    sig_out = static_cast<SAMPLETYPE *>(malloc(sig_out_size * nChannels * sizeof(SAMPLETYPE)));
    if(!sig_out) return NULL;
    
    do{
        remains = (sig_in_len*nChannels - sig_in_pos);
        if(remains >= bufferSize) num = bufferSize;
        else num = remains;
        
        int nSamples = num/nChannels;
        st.putSamples(sig_in + sig_in_pos, nSamples);

        sig_in_pos += num; 
        do{
            //TODO: use vector instead of malloc?
            if(sig_out_pos + bufferSize > sig_out_size){
                sig_out_size = sig_out_pos + bufferSize;
                float ratio;
                if(sig_in_pos == 0) {
                    ratio = 1;
                } else {
                    ratio = static_cast<float>(1.1*sig_out_pos / sig_in_pos);
                    //if(ratio < 1) ratio = 1;//ok?
                }
                sig_out_size += static_cast<int>(ratio*(sig_in_len*nChannels - sig_in_pos));
                void *tmp = realloc(static_cast<void *>(sig_out), sig_out_size*sizeof(SAMPLETYPE));
                if(!tmp){
                    if(sig_out_size!= 0) free(sig_out);
                    sig_out = NULL;
                }else{
                    sig_out = static_cast<SAMPLETYPE *>(tmp);
                }
            }
            if(!sig_out) return NULL;
            nSamples = st.receiveSamples(sig_out + sig_out_pos, buffSizeSamples);
            sig_out_pos += nSamples*nChannels;
        }while(nSamples > 0);
    }while(sig_in_pos < sig_in_len*nChannels);

    st.flush();
    int nSamples;
        do{
            if(sig_out_pos + bufferSize > sig_out_size){
                sig_out_size = sig_out_pos + bufferSize*(1 + 10);
                void *tmp = realloc(static_cast<void *>(sig_out), sig_out_size*sizeof(SAMPLETYPE));
                if(!tmp){
                    if(sig_out_size!= 0) free(sig_out);
                    sig_out = NULL;
                }else{
                    sig_out = static_cast<SAMPLETYPE *>(tmp);
                }
            }
            if(!sig_out) return NULL;
            nSamples = st.receiveSamples(sig_out + sig_out_pos, buffSizeSamples);
            sig_out_pos += nSamples*nChannels;
        }while(nSamples > 0);


    assert(sig_in_pos == sig_in_len*nChannels);
    *sig_out_len = sig_out_pos/nChannels;
    return sig_out;
}

/* The wrapper */
double *
shiftPitch_double(
        double *sig_in, int sig_in_len, int nChannels, 
        int fs, float alpha, int quick, int aa, 
        int bufferSize, 
        int *sig_out_len
        )
{
    //double -> SAMPLETYPE
    SAMPLETYPE *sig_in_internal = new SAMPLETYPE[sig_in_len*nChannels];
    for(int i = 0; i < sig_in_len; i++){
        int pos = i*nChannels;
        for(int channel = 0; channel < nChannels; channel++){
            sig_in_internal[pos+channel] = static_cast<SAMPLETYPE>(sig_in[pos+channel]);
        }
    }
    SAMPLETYPE *sig_out_internal = shiftPitch(sig_in_internal, 
            sig_in_len, nChannels,
            fs, alpha, quick, aa, bufferSize, sig_out_len);
    delete [] sig_in_internal;
    double *sig_out = new double[(*sig_out_len)*nChannels];
    //SAMPLETYPE -> double
    for(int i = 0; i < *sig_out_len; i++){
        int pos = i*nChannels;
        for(int channel = 0; channel < nChannels; channel++){
            sig_out[pos + channel] = static_cast<double>(sig_out_internal[pos + channel]);
        }
    }
    delete [] sig_out_internal;
    return sig_out;
}

void shiftPitch_release(double *sig_out)
{
    if(sig_out) free(sig_out);
}
