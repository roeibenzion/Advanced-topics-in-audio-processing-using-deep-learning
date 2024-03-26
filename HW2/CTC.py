import numpy as np
import matplotlib.pyplot as plt


def CTC_B_function(output, blank_token=0):
    B_output = ""
    i = 0
    while output[i] == blank_token:
        i += 1
    if i < len(output):
        B_output += output[i]
    
    for j in range(i+1, len(output)):
        if output[j] == blank_token:
            continue
        elif output[j] == output[j-1]:
            continue
        else:
            B_output += output[j]
    return B_output

def forward_pass(z, P, char_to_num):
    """
    Computes the forward pass of the CTC algorithm.

    Args:
    - z (list): padded input
    - P (list): Output probabilities for the input sequence (|Y| X T)
    - char_to_num (dict): Dictionary mapping characters to their numeric representation.

    Returns:
    - alpha (numpy.ndarray): Forward probabilities.
    """
    _, T = P.shape
    S = len(z)
    alpha = np.zeros(shape=(S, T))
    alpha[0,0] = P[char_to_num['^'], 0]
    alpha[1,0] = P[char_to_num[z[1]], 0]
    for t in range(1, T):
        for i in range(0, S):
            s = char_to_num[z[i]]
            if s == char_to_num['^'] or (i >= 2 and z[i] == z[i-2]):
                alpha[i,t] = (alpha[i, t-1] + alpha[i-1, t-1]) * P[s, t]
            else:
                alpha[i,t] = (alpha[i, t-1] + alpha[i-1, t-1] + alpha[i-2, t-1]) * P[s, t]
    return alpha

def plot_forward_probabilities(alpha, char_to_num, z):
    S, T = alpha.shape
    labels = list(z)

    plt.figure(figsize=(10, 6))
    plt.imshow(alpha, cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar(label='Forward Probability')
    plt.xlabel('Time Step')
    plt.ylabel('Character')
    plt.xticks(np.arange(T), np.arange(T) + 1)
    plt.yticks(np.arange(S), labels)
    plt.title('Forward Probabilities')
    plt.grid(False)
    plt.show()

def ctc_force_align(z, P, char_to_num):
    _, T = P.shape
    S = len(z)
    alpha = np.zeros(shape=(S, T))
    backtrace = np.zeros_like(alpha)
    alpha[0,0] = P[char_to_num['^'], 0]
    alpha[1,0] = P[char_to_num[z[1]], 0]
    for t in range(1, T):
        for i in range(0, S):
            s = char_to_num[z[i]]
            if s == char_to_num['^'] or (i >= 2 and z[i] == z[i-2]):
                alpha[i,t] = max(alpha[i, t-1], alpha[i-1, t-1]) * P[s, t]
                m = np.argmax([alpha[i, t-1], alpha[i-1, t-1]])
                backtrace[i, t] = i if m == 0 else i-1
            else:
                alpha[i,t] = max(alpha[i, t-1], alpha[i-1, t-1], alpha[i-2, t-1]) * P[s, t]
                m = np.argmax([alpha[i, t-1], alpha[i-1, t-1], alpha[i-2, t-1]])
                backtrace[i, t] = i if m == 0 else (i-1 if m == 1 else i-2)
    
    c = np.max(alpha[:, -1])
    d = z[np.argmax(alpha[:,-1])]
    return alpha, backtrace, c, d

def get_most_probable_path(backtrace, alpha, z):
    _, T= backtrace.shape  # Corrected the order of dimensions
    
    path = []

    # Find the index of the highest probability in the last column of alpha
    max_index = np.argmax(alpha[:, -1])
    
    # Starting from the most probable end token, trace back using backtrace matrix
    i = max_index
    for t in range(T-1, 0, -1):
        path.insert(0, z[i])  # Insert at the beginning to maintain order
        prev_index = int(backtrace[i, t])  # Corrected the order of indices
        i = prev_index

    # Insert the starting token
    path.insert(0, z[i])

    return ''.join(path)


pred = np.array([
    [0.8, 0.2, 0.0],
    [0.2, 0.8, 0.0],
    [0.3, 0.5, 0.2],
    [0.7, 0.1, 0.2],
    [0.0, 0.0, 1.0]
])

pred = pred.T
#log_pred = np.log(pred)


alpha_bet_map = {0: 'a', 1: 'b', 2: '^'}
char_to_num = {'a': 0, 'b': 1, '^': 2}
y = 'aba'
z = '^a^b^a^'
alpha = forward_pass(z, pred, char_to_num)
print(alpha.shape)
plot_forward_probabilities(alpha, alpha_bet_map, z)

# Define input and character-to-number mapping
z = '^a^b^a^'
char_to_num = {'^': 0, 'a': 1, 'b': 2}

# Get the index for 'a', 'b', and the end character '^'
a_idx = char_to_num['a']
b_idx = char_to_num['b']
end_idx = char_to_num['^']

# Calculate the probability of 'aba'
aba_prob = alpha[-2, 4] + alpha[-1, 4]

print("Probability of 'aba':", aba_prob)

alpha, backtrace, c, d = ctc_force_align(z, pred, char_to_num)
#plot_forward_probabilities(alpha, char_to_num, z)
print(get_most_probable_path(backtrace, alpha, z))
prod = 1.0
for i in range(5):
    prod *= np.max(alpha[:,i])
print(prod)


    
