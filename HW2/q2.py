# q2 plot

import librosa
import librosa.display as display
import numpy as np
import matplotlib.pyplot as plt

window_size = 0.025
hop_size = 0.010
n_mels = 80
sample_rate = 16000

audio_files = ['0.wav', '1.wav', '2.wav', '3.wav', '4.wav',
               '5.wav', '6.wav', '7.wav', '8.wav', '9.wav']

fig, axes = plt.subplots(5, 2, figsize=(20, 25))
axes = axes.flatten()

for i, audio in enumerate(audio_files):
    y, sr = librosa.load(audio, sr=sample_rate)
    n_fft = int(window_size * sr)
    hop_length = int(hop_size * sr)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    img = display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', ax=axes[i])
    axes[i].set_title(f'Mel Spectrogram of {audio}')
    axes[i].set_xlabel('Time [s]')
    axes[i].set_ylabel('Mel Frequency')


fig.colorbar(img, ax=axes, orientation='vertical', fraction=.02)
plt.tight_layout()
plt.savefig('woman2.png', dpi=300)
plt.show()
