from pycbc.frame import read_frame
import urllib.request
import pylab
import pycbc
from pycbc.catalog import Merger
from pycbc.filter import resample_to_delta_t, highpass
from pycbc.filter import matched_filter
from pycbc.waveform import get_td_waveform
from pycbc.filter import sigma
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.types import TimeSeries
import numpy
%matplotlib inline

def get_file(fname):
    url = "https://github.com/aman8533/GravitationalWaves/raw/main/GWChallenge/{}"
    url = url.format(fname)
    urllib.request.urlretrieve(url, fname)
    print('Getting : {}'.format(url))

files = ['challenge2.gwf']
for fname in files:
    get_file(fname)

file_name = "challenge2.gwf"

channel_name = "H1:CHALLENGE2"

ts = read_frame(file_name, channel_name)
samplerate = ts.get_sample_rate()
duration = ts.get_duration()

strain = highpass(ts, 20)
strain = resample_to_delta_t(strain, 1.0/2048)

#Plot the data in the time-domain.
conditioned = strain.crop(2, 2)
psd = conditioned.psd(4)


psd = interpolate(psd, conditioned.delta_f)

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

#pylab.loglog(psd.sample_frequencies, psd)
#pylab.ylabel('$Strain^2 / Hz$')
#pylab.xlabel('Frequency (Hz)')
#pylab.xlim(30, 1024)

#ploting the strain data
pylab.plot(conditioned.sample_times+tshiftgaussian , conditioned)
pylab.xlabel('Time (s)')
pylab.show()
# Strained data plotting ends

#Generated waveform with mass m = 30 and spin = 0
m = 30 # Solar masses
hp, hc = get_td_waveform(approximant="SEOBNRv4_opt",
                    mass1=m,
                    mass2=m,
                    delta_t=conditioned.delta_t,
                    f_lower=20)

# Resized the vector to match our data
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

# Removed time corrupted by the template filter and the psd filter
# removed 4 seconds at the beginning and end for the PSD filtering
# removed 4 additional seconds at the beginning to account for

snr = snr.crop(4 + 4, 4)


pylab.figure(figsize=[10, 4])
pylab.plot(snr.sample_times, abs(snr))
pylab.ylabel('Signal-to-noise')
pylab.xlabel('Time (s)')
pylab.show()

#Calculate SnR Peak
peak = abs(snr).numpy().argmax()
snrp = snr[peak]
time = snr.sample_times[peak]

print("We found a signal at {}s with SNR {}".format(time+tshiftgaussian, 
                                                    abs(snrp)))


# Shifted the template to the peak time
dt = time - conditioned.start_time
aligned = template.cyclic_time_shift(dt)

# scaled the template so that it would have SNR 1 in this data
aligned /= sigma(aligned, psd=psd, low_frequency_cutoff=20.0)

# Scaled the template amplitude and phase to the peak value
aligned = (aligned.to_frequencyseries() * snrp).to_timeseries()
aligned.start_time = conditioned.start_time

#Whitened the data

white_data = (conditioned.to_frequencyseries() / psd**0.5).to_timeseries()
white_template = (aligned.to_frequencyseries() / psd**0.5).to_timeseries()

white_data = white_data.highpass_fir(30., 512).lowpass_fir(300, 512)
white_template = white_template.highpass_fir(30, 512).lowpass_fir(300, 512)

# Selected the time around the merger
mergertime = time+tshiftgaussian
white_data = white_data.time_slice(mergertime-.2, mergertime+.1)
white_template = white_template.time_slice(mergertime-.2, mergertime+.1)

pylab.figure(figsize=[15, 3])
pylab.plot(white_data.sample_times, white_data, label="Data")
pylab.plot(white_template.sample_times, white_template, label="Template")
pylab.legend()
pylab.show()

subtracted = conditioned - aligned

# Plotted the original data and the subtracted signal data

# Plotted a spectrogram (or q-transform) of the data, and try to identify the signal

for data, title in [(conditioned, 'Original H1 Data'),
                    (subtracted, 'Signal Subtracted from H1 Data')]:

    t, f, p = data.whiten(4, 4).qtransform(.001, logfsteps=100, qrange=(8, 8), frange=(20, 512))
    pylab.figure(figsize=[15, 3])
    pylab.title(title)
    pylab.pcolormesh(t+tshiftgaussian, f, p**0.5, vmin=1, vmax=6, shading='auto')
    pylab.yscale('log')
    pylab.xlabel('Time (s)')
    pylab.ylabel('Frequency (Hz)')
    #pylab.xlim(mergertime - 2, mergertime + 1)
    pylab.show()
