import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import os
from scipy.io import loadmat
from seqNMF import seq_nmf, plot

data = loadmat(os.path.join('..', 'MackeviciusData.mat'))
W, H, cost, loadings, power = seq_nmf(data['NEURAL'])

h = plot(W, H)
h.show()

sns.heatmap(data)
plt.show()

