import librosa
import librosa.display as display
import scipy.signal as signal
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import parselmouth
from scipy.interpolate import interp1d

def trim_silence(y, threshold):
    # Trim the beginning and ending silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=threshold)
    return y_trimmed

def downsample_method1(y):
    # This method takes every even sample
    # From 32kHz to 16kHz
    return y[::2]

def downsample_method2(y, sr, new_sr):
    num_samples_new = int(len(y) * new_sr / sr)
    # Resample the signal to the new length
    return signal.resample(y, num_samples_new)


def draw_pitch(pitch, ax, n_frames):
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values==0] = np.nan
    # Create interpolation function
    f = interp1d(pitch.xs(), pitch_values, kind='linear', fill_value='extrapolate')
    # Interpolate pitch values to match spectrogram frames
    new_xs = np.linspace(pitch.xmin, pitch.xmax, n_frames)
    intepulated_pitch_values = f(new_xs)
    # Plot pitch contour
    ax.plot(new_xs, intepulated_pitch_values, color='r', linewidth=2)

def plot_wave(y, sr, ax):
    display.waveshow(y, sr=sr, ax=ax)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude')

def plot_spectrogram(y, S, sr, n_fft, hop_length, ax):
    # Spectrogram
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    # Pitch contour using Praat
    snd = parselmouth.Sound(y.T, sampling_frequency=sr)
    pitch = snd.to_pitch()  
    draw_pitch(pitch, ax, n_frames=sr//hop_length)
    # Add labels to axises
    display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', ax=ax)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')

def draw_mel_spectrogram(y, S, sr, n_fft, hop_length, ax):
    # Mel spectrogram
    S_mel = librosa.feature.melspectrogram(S=S, sr=sr, n_fft=n_fft, hop_length=hop_length)
    S_db_mel = librosa.amplitude_to_db(S_mel, ref=np.max)
    display.specshow(S_db_mel, sr=sr, hop_length=hop_length, x_axis= 'time', y_axis='mel', ax=ax)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')

def draw_energy_rms(y, sr, n_fft, hop_length, ax):
    
    # Energy and RMS
    frame_length = n_fft
    rms = np.array([
        np.sqrt(  
            (1 / frame_length) * sum(y[i:i + frame_length] ** 2)
        ) 
        for i in range(0, len(y), hop_length)
    ])
    energy = np.array([
    sum(abs(y[i:i+frame_length]**2))
    for i in range(0, len(y), hop_length)
                                    ]) 
    
    # normalize
    energy = energy / np.max(energy)
    rms = rms / np.max(rms)
    # Calculate time in seconds
    frame_times = np.arange(len(rms)) * hop_length / sr

    # Plotting
    ax.plot(frame_times, rms, label='RMS') 
    ax.plot(frame_times, energy, label='Energy')

    # Labels
    ax.set_xlabel('Time (seconds)')  
    ax.set_ylabel('Normalized Amplitude')
    
    # Legend
    ax.legend()

    
def plot_spectrogram_and_save(y, sr, n_fft, hop_length, filename):
    S = librosa.stft(y, n_fft=n_fft, win_length=n_fft, hop_length=hop_length)
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_spectrogram(y, S, sr, n_fft, hop_length, ax)
    plt.savefig(filename)
    plt.close()

def multi_plot(y, sr):
    """
    1. D
    Assuming it is already loaded with librosa.
    """
    window_size = 0.020  # 20ms window (in seconds)
    hop_size = 0.010     # 10ms hop (in seconds)
    n_fft = int(sr * window_size)
    hop_length = int(sr * hop_size)
    fig, ax = plt.subplots(nrows=4, figsize=(20, 10))
    # Audio
    plot_wave(y, sr, ax[0])
    # Spectrogram
    S = librosa.stft(y, n_fft=n_fft, win_length=n_fft, hop_length=hop_length)
    plot_spectrogram(y, S, sr, n_fft, hop_length, ax[1])
    plot_spectrogram_and_save(y, sr, n_fft, hop_length, 'spectrogram.png')
    # Mel spectrogram
    draw_mel_spectrogram(y, S, sr, n_fft, hop_length, ax[2])
    # Energy and RMS
    draw_energy_rms(y, sr, n_fft, hop_length, ax[3])
    plt.show()
    

def q_1B(y, sr, path):
    new_sr = 32000
    num_samples_new = int(len(y) * new_sr / sr)
    y_resampled = signal.resample(y, num=num_samples_new)
    sf.write(path, y_resampled, new_sr)

def q_1C(y, sr, path):
    new_sr = 16000
    y = downsample_method1(y)
    sf.write(path + '_even.wav', y, new_sr)
    num_samples_new = int(len(y) * new_sr / sr)
    y = signal.resample(y, num=num_samples_new)
    sf.write(path + '_resample.wav', y, new_sr)

def q_1D(y, sr):
    y = y.astype(np.float32)
    multi_plot(y, sr)
    # The missing timeframes in the pitch contour 
    # is because there is no voice in that timeframe

def main():
    sr = 44100
    # Load the audio files
    y1, _ = librosa.load('20CM.wav', sr=sr)
    y2, _ = librosa.load('3M.wav', sr=sr)
    # Trim the silence and update y2
    y2 = trim_silence(y2, 20)
    sf.write('3M_trimmed.wav', y2, sr)
    # 1.B
    q_1B(y1, sr, '20CM_32k.wav')
    q_1B(y2, sr, '3M_trimmed_32k.wav')
    # 1.C
    q_1C(y1, sr, '20CM')
    q_1C(y2, sr, '3M_trimmed')

if __name__ == '__main__':
    main()