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

# Path to the folders
training_set_folder = 'training_set'
#training_set_folder = 'evaluation_set'
class_representive_folder = 'class_representative'

# Load the training set audio files
speakers = ['Gari', 'Koral', 'Sheli', 'Ron']
#speakers = ['David', 'Evgeny', 'Victoria', 'Lihi']
training_set = []
for speaker in speakers:
    speaker_folder = os.path.join(training_set_folder, speaker)
    speaker_files = []
    for i in range(10):
        file_path = os.path.join(speaker_folder, f"{i}.wav")
        speaker_files.append(file_path)
    training_set.append(speaker_files)

training_set_flat = [item for sublist in training_set for item in sublist]

# Load the class representative audio files
class_representative_files = []
class_representative_folder = os.path.join(class_representive_folder, 'Tamir')
for i in range(10):
    file_path = os.path.join(class_representative_folder, f"{i}.wav")
    class_representative_files.append(file_path)

distance_matrix = np.zeros((len(training_set_flat), len(class_representative_files)))

dtw_distances_file_path = 'dtw_distances.txt'
with open(dtw_distances_file_path, 'w') as file:
    file.write('TrainingFile,DBFile,DTWDistance\n')  # Header

for i, digit_file in enumerate(training_set_flat):
    digit_audio, _ = librosa.load(digit_file, sr=16000)
    for j, db_file in enumerate(class_representative_files):
        db_audio, _ = librosa.load(db_file, sr=16000)
        dtw_distance = dtw(digit_audio, db_audio)
        distance_matrix[i, j] = dtw_distance
        
        with open(dtw_distances_file_path, 'a') as file:
            file.write(f"{digit_file},{db_file},{dtw_distance}\n")
    
    # Print progress every 10 iterations
    if (i + 1) % 10 == 0:
        print(f"Progress: {i + 1}/{len(training_set_flat)} audio files processed.")

print("Distance Matrix:")
print(distance_matrix)
