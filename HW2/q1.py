# for q1

import librosa
import soundfile as sf


def q1(audio,path):
  y, sr = librosa.load(audio, sr=16000)
  sf.write(path, y, sr)


q1('0.wav','0.wav')
q1('1.wav','1.wav')
q1('2.wav','2.wav')
q1('3.wav','3.wav')
q1('4.wav','4.wav')
q1('5.wav','5.wav')
q1('6.wav','6.wav')
q1('7.wav','7.wav')
q1('8.wav','8.wav')
q1('9.wav','9.wav')



