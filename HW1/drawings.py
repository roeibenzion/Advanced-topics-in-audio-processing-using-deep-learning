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
''' 
num_samples = 500
rect = np.zeros(num_samples)
rect[150:350] = 1
rect_signal = signal(rect, np.linspace(-50,50, num=num_samples), 'Rectangular Signal')
rect_signal.draw()
'''

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

def plot_function():
    # Define the range of omega values
    omega = np.linspace(-10, 10, 1000)

    # Define the sinc function
    sinc = lambda x: np.sinc(x / np.pi)

    # Calculate the function values for each omega
    function_values = np.sum(np.pi * sinc(omega * np.pi / 2) * np.exp(-1j * omega * 2 * np.pi * np.arange(-10, 11)[:, np.newaxis]), axis=0)

    # Plot the real and imaginary parts of the function
    plt.figure(figsize=(10, 6))
    plt.plot(omega, np.real(function_values), label='Real part')
    plt.plot(omega, np.imag(function_values), label='Imaginary part')
    plt.xlabel('Omega')
    plt.ylabel('Function Value')
    plt.title('Plot of the Function')
    plt.legend()
    plt.grid(True)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_magnitude_and_phase(w_range):
    # Generate frequencies
    w = np.linspace(w_range[0], w_range[1], 1000)

    # Calculate the magnitude and phase
    magnitude = 1 / np.sqrt(1 + w**2)
    phase = -np.arctan(w)

    # Plot magnitude
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(w, magnitude)
    plt.title('Magnitude Response')
    plt.xlabel('Frequency (ω)')
    plt.ylabel('Magnitude')
    plt.grid(True)

    # Plot phase
    plt.subplot(2, 1, 2)
    plt.plot(w, phase)
    plt.title('Phase Response')
    plt.xlabel('Frequency (ω)')
    plt.ylabel('Phase (radians)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Example usage with frequency range from 0 to 10
plot_magnitude_and_phase([0, 10])




