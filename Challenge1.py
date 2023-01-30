from pycbc.frame import read_frame
import urllib.request
import pylab
from pycbc.catalog import Merger
from pycbc.filter import resample_to_delta_t, highpass
from pycbc.filter import matched_filter
import numpy
%matplotlib inline

def get_file(fname):
  url = "https://github.com/aman8533/GravitationalWaves/raw/main/GWChallenge/{}"
  url = url.format(fname)
  urllib.request.urlretrieve(url, fname)
  print('Getting : {}'.format(url))

files = ['challenge1.gwf']
for fname in files:
  get_file(fname)
                

file_name = "challenge1.gwf"

channel_name = "H1:CHALLENGE1"

ts = read_frame(file_name, channel_name)
samplerate = ts.get_sample_rate()
duration = ts.get_duration()
print(ts)
print("Sample Rate of the dataset",samplerate)
print("Duration of the dataset",duration)

strain = highpass(ts, 15.0)
strain = resample_to_delta_t(strain, 1.0/2048)

#Plotted the data in the time-domain.
conditioned = strain.crop(2, 2)
psd = conditioned.psd(4)
psd = interpolate(psd, conditioned.delta_f)
psd = inverse_spectrum_truncation(psd, int(4 * conditioned.sample_rate), low_frequency_cutoff=15)
tshiftgaussian = 60
pylab.plot(conditioned.sample_times+tshiftgaussian, conditioned)
pylab.xlabel('Time (s)')
pylab.show()

print("Strain Sampletimes",conditioned.sample_times)


# Plotted a spectrogram (or q-transform) of the data, and try to identify the signal

for data in [conditioned]:
    t, f, p = data.whiten(4, 4).qtransform(.001, logfsteps=100, qrange=(8, 8), frange=(20, 512))
    pylab.figure(figsize=[15, 3])
    pylab.title(title)
    pylab.pcolormesh(t+tshiftgaussian, f, p**0.5, vmin=1, vmax=6, shading='auto')
    pylab.yscale('log')
    pylab.xlabel('Time (s)')
    pylab.ylabel('Frequency (Hz)')

    pylab.show()
    print ("Merger time is observed at:",tshiftgaussian,"seconds")
    # GW170814 data
    merger = Merger("GW170814")
    # Got the data from the Hanford detector
    strain = merger.strain('H1')
    print("Merger Time of GW170814 is:",merger.time)
    # Removed the low frequency content and downsample the data to 2048Hz
    strain = highpass(strain, 15.0)
    strain = resample_to_delta_t(strain, 1.0/2048)
    conditioned = strain.crop(2, 2)

    pylab.plot(conditioned.sample_times, conditioned)
    pylab.xlabel('Time (s)')
    pylab.show()
