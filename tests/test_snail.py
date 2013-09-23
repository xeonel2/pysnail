import snail
import numpy as np

if __name__ == "__main__":
    import sys

    x,fs = signal.wavread("clean.wav")

    buf_size = int(sys.argv[1])
    alpha = float(sys.argv[2])
    y = snail.freqcomp_mono(x, fs, buf_size, alpha)

    signal.wavwrite(y, fs, ("%s.wav" % (sys.argv[3])))