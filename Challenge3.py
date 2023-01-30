from pycbc.frame import read_frame
import urllib.request
import pylab
import pycbc
from pycbc import frame
from pycbc.catalog import Merger
from pycbc.filter import resample_to_delta_t, highpass
from pycbc.filter import matched_filter
from pycbc.waveform import get_td_waveform
from pycbc.filter import sigma
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.types import TimeSeries
import numpy
%matplotlib inline
 
# -- download data
! wget https://www.gw-openscience.org/s/workshop3/challenge/challenge3.gwf
 
# -- for pycbc
 
urllib.request.urlretrieve("https://www.gw-openscience.org/s/workshop3/challenge/challenge3.gwf", "challenge3.gwf")
pycbc_strain = read_frame('challenge3.gwf', 'H1:CHALLENGE3')
print(pycbc_strain)
 
pycbc_strain = highpass(pycbc_strain, 20)
pycbc_strain = resample_to_delta_t(pycbc_strain, 1.0/2048)
 
#Plot the data in the time-domain.
conditioned = pycbc_strain.crop(2, 2)
psd = conditioned.psd(4)
 
# Now that we have the psd we need to interpolate it to match our data
# and then limit the filter length of 1 / PSD. After this, we can
# directly use this PSD to filter the data in a controlled manner
 
psd = interpolate(psd, conditioned.delta_f)
 
# 1/PSD will now act as a filter with an effective length of 4 seconds
# Since the data has been highpassed above 20 Hz, and will have low values
# below this we need to informat the function to not include frequencies
# below this frequency. 
psd = inverse_spectrum_truncation(psd, int(4 * conditioned.sample_rate),
                                  low_frequency_cutoff=20)
 
print(psd.sample_frequencies)
pylab.plot(psd.sample_frequencies, psd, label="Spectral Data")
 
pylab.yscale('log')
pylab.xscale('log')
pylab.ylim(1e-47, 1e-41)
pylab.xlim(20, 1024)
pylab.ylabel('$Strain^2 / Hz$')
pylab.xlabel('Frequency (Hz)')
pylab.grid()
pylab.legend()
pylab.show()
 
#PSD Plotting ends
 
#plot the strain data
pylab.plot(conditioned.sample_times , conditioned)
pylab.xlabel('Time (s)')
pylab.show()
# Strain data plotting ends
 
#Generate waveform with mass m = 30 and spin = 0
m = 10 # Solar masses
hp, hc = get_td_waveform(approximant="SEOBNRv4_opt",
                     mass1=m,
                     mass2=m,
                     delta_t=conditioned.delta_t,
                     f_lower=20)
 
# Resize the vector to match our data
hp.resize(len(conditioned))
 
pylab.figure()
pylab.title('Before shifting')
pylab.plot(hp.sample_times, hp)
pylab.xlabel('Time (s)')
pylab.ylabel('Strain')
 
template = hp.cyclic_time_shift(hp.start_time)
 
pylab.figure()
pylab.title('After shifting')
pylab.plot(template.sample_times, template)
pylab.xlabel('Time (s)')
pylab.ylabel('Strain')
 
#Wave form generation ends 
 
#Create Matched filter
snr = matched_filter(template, conditioned,
                     psd=psd, low_frequency_cutoff=20)
 
# Remove time corrupted by the template filter and the psd filter
# We remove 4 seconds at the beginning and end for the PSD filtering
# And we remove 4 additional seconds at the beginning to account for
# the template length (this is somewhat generous for 
# so short a template). A longer signal such as from a BNS, would 
# require much more padding at the beginning of the vector.
snr = snr.crop(4 + 4, 4)
 
# Why are we taking an abs() here?
# The `matched_filter` function actually returns a 'complex' SNR.
# What that means is that the real portion correponds to the SNR
# associated with directly filtering the template with the data.
# The imaginary portion corresponds to filtering with a template that
# is 90 degrees out of phase. Since the phase of a signal may be 
# anything, we choose to maximize over the phase of the signal.
pylab.figure(figsize=[10, 4])
pylab.plot(snr.sample_times, abs(snr))
pylab.ylabel('Signal-to-noise')
pylab.xlabel('Time (s)')
pylab.show()
 
#Calculate SnR Peak
peak = abs(snr).numpy().argmax()
snrp = snr[peak]
time = snr.sample_times[peak]
 
print("We found a signal at {}s with SNR {}".format(time, abs(snrp)))
 
# The time, amplitude, and phase of the SNR peak tell us how to align
# our proposed signal with the data.
 
# Shift the template to the peak time
dt = time - conditioned.start_time
aligned = template.cyclic_time_shift(dt)
 
# scale the template so that it would have SNR 1 in this data
aligned /= sigma(aligned, psd=psd, low_frequency_cutoff=20.0)
 
# Scale the template amplitude and phase to the peak value
aligned = (aligned.to_frequencyseries() * snrp).to_timeseries()
aligned.start_time = conditioned.start_time
 
#Whiten the data
# We do it this way so that we can whiten both the template and the data
white_data = (conditioned.to_frequencyseries() / psd**0.5).to_timeseries()
white_template = (aligned.to_frequencyseries() / psd**0.5).to_timeseries()
 
white_data = white_data.highpass_fir(30., 512).lowpass_fir(300, 512)
white_template = white_template.highpass_fir(30, 512).lowpass_fir(300, 512)
 
# Select the time around the merger
mergertime = time
white_data = white_data.time_slice(mergertime-.2, mergertime+.1)
white_template = white_template.time_slice(mergertime-.2, mergertime+.1)
 
pylab.figure(figsize=[15, 3])
pylab.plot(white_data.sample_times, white_data, label="Data")
pylab.plot(white_template.sample_times, white_template, label="Template")
pylab.legend()
pylab.show()
 
subtracted = conditioned - aligned
 
# Plot the original data and the subtracted signal data
 
#Qtransform white data and plot the graph
#Qtransform white data and plot the graph[
# Plot a spectrogram (or q-transform) of the data, and try to identify the signal
 
for data, title in [(conditioned, 'Original H1 Data'),
                    (subtracted, 'Signal Subtracted from H1 Data')]:
 
    t, f, p = data.whiten(4, 4).qtransform(.001, logfsteps=100, qrange=(8, 8), frange=(20, 512))
    pylab.figure(figsize=[15, 3])
    pylab.title(title)
    pylab.pcolormesh(t, f, p**0.5, vmin=1, vmax=6, shading='auto')
    pylab.yscale('log')
    pylab.xlabel('Time (s)')
    pylab.ylabel('Frequency (Hz)')
    #pylab.xlim(mergertime - 2, mergertime + 1)
    pylab.show()



[-3.04517369e-19 -3.35465069e-19 -3.35210566e-19 ...  4.46471052e-19
  5.07624678e-19  4.98623016e-19]
[0.00000000e+00 2.44379277e-04 4.88758553e-04 ... 1.02399951e+03
 1.02399976e+03 1.02400000e+03]
