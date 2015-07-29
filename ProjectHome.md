A python package containing pitch-related functions.

Current functions include:

**get\_pitch**
:   Estimates the pitch of a signal. Uses the Yin algorithm, a well-established autocorrelation based pitch algorith. Read a paper on Yin here: http://audition.ens.fr/adc/pdf/2002_JASA_YIN.pdf

**shift\_pitch**
:   Applies a specified amount of freq compression/expansion to a signal. Uses (and depends on) the soundtouch library: http://www.surina.net/soundtouch/