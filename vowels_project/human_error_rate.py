import numpy as np

f = open('human_classification.txt')
human_classification_rate = np.average([float(line.strip().split()[1]) for line in f.readlines()[1:]]) / 100
print('Human error rate: ' + str(100 * (1 - human_classification_rate)) + '%')
