'''
This is a python file that contains functions to draw a signal.
'''

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def set_style():
    sns.set_theme()
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

class signal: 
    def __init__(self, signal, lin_space, title):
        self.signal = signal
        self.lin_space = lin_space
        self.title = title
    
    def to_time_domain(self):
        self.signal = np.fft.ifft(self.signal)
        self.lin_space = np.arange(len(self.signal))
        self.title = 'Time Domain'

    def to_frequency_domain(self):
        self.signal = np.fft.fft(self.signal)
        self.lin_space = np.fft.fftfreq(len(self.signal), d=1)
        self.title = 'Frequency Domain'
        self.time = False
    
    def get_magnitude(self):
       assert self.time == False, 'Signal is in time domain'
       return np.abs(self.signal)

    def get_angle(self):
        assert self.time == False, 'Signal is in time domain'
        self.real = np.real(self.signal)
        self.imag = np.imag(self.signal)
        phase_angle = np.arctan(self.imag/self.real)
        return phase_angle
    
    def draw(self):
        set_style()
        plt.plot(self.lin_space, self.signal)
        plt.title(self.title)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.show()

        plt.magnitude_spectrum(self.signal, Fs=1, sides='twosided')
        plt.show()

        plt.phase_spectrum(self.signal, Fs=1, sides='twosided')
        plt.show()

        
        
# Q1 - e - ii
        
num_samples = 500
rect = np.zeros(num_samples)
rect[150:350] = 1
rect_signal = signal(rect, np.linspace(-50,50, num=num_samples), 'Rectangular Signal')
rect_signal.draw()


'''
# Q2 - a - iii

def cn(n, w0, t):
    return np.exp(1j * n * w0 * t)

def f_unit_impulse_train(t, T, low, high):
    return (1/T)*np.sum([cn(n, 2 * np.pi / T, t) for n in range(low, high + 1)])
num_samples = 1000

unit_impluse_train = np.zeros(num_samples)
unit_impluse_train[50:1000:50] = 1
unit_impluse_train_signal = signal(unit_impluse_train, np.linspace(-50,50, num=num_samples), 'Unit Impulse Train')
unit_impluse_train_signal.draw()
# Set the style
set_style()
plt.plot(np.linspace(-50, 50, num=100),[f_unit_impulse_train(t, 2*np.pi, -10000, 10000) for t in range(-50, 50)])
plt.show()
'''
