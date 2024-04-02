import os
from numba import jit
import numpy as np
import librosa

@jit(nopython=True)
def dtw(vec1, vec2):
    len1, len2 = len(vec1), len(vec2)
    distances = np.zeros((len1, len2))
    
    for i in range(len1):
        for j in range(len2):
            distances[i, j] = (vec1[i] - vec2[j]) ** 2

    dtw_cost = np.full((len1 + 1, len2 + 1), np.inf)
    dtw_cost[0, 0] = 0

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = distances[i - 1, j - 1]
            dtw_cost[i, j] = cost + min(dtw_cost[i - 1, j], dtw_cost[i, j - 1], dtw_cost[i - 1, j - 1])

    return dtw_cost[len1, len2]

#training_set_folder = 'training_set'
training_set_folder = 'evaluation_set'
class_representative_folder = 'class_reprentative'

#speakers = ['Gari', 'Sheli', 'Koral', 'Ron']
speakers = ['David', 'Evgeny', 'Victoria', 'Lihi']
training_set = []
for speaker in speakers:
    speaker_folder = os.path.join(training_set_folder, speaker)
    speaker_files = []
    for i in range(10):
        file_path = os.path.join(speaker_folder, f"{i}.wav")
        speaker_files.append(file_path)
    training_set.append(speaker_files)

training_set_flat = [item for sublist in training_set for item in sublist]
class_representative_files = []
for i in range(10):
    file_path = os.path.join(class_representative_folder, 'Tamir', f"{i}.wav")
    class_representative_files.append(file_path)

distance_matrix = np.zeros((len(training_set_flat), len(class_representative_files)))

dtw_distances_file_path = 'AGC_dtw_distances_normalized_evaluation_set.txt'
with open(dtw_distances_file_path, 'w') as file:
    file.write('TrainingFile,DBFile,DTWDistance\n')

for i, digit_file in enumerate(training_set_flat):
    digit_audio, sr_digit = librosa.load(digit_file, sr=16000)
    digit_length = len(digit_audio) / sr_digit
    for j, db_file in enumerate(class_representative_files):
        db_audio, sr_db = librosa.load(db_file, sr=16000)
        db_length = len(db_audio) / sr_db
        dtw_distance = dtw(digit_audio, db_audio)
        normalized_distance = dtw_distance / (digit_length + db_length)
        distance_matrix[i, j] = normalized_distance
        with open(dtw_distances_file_path, 'a') as file:
            file.write(f"{digit_file},{db_file},{normalized_distance}\n")
    
    if (i + 1) % 10 == 0:
        print(f"Progress: {i + 1}/{len(training_set_flat)} audio files processed.")

print("Normalized Distance Matrix:")
print(distance_matrix)
