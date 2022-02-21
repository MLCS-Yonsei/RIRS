from sklearn import naive_bayes, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold as rskf
import pandas as pd
from scipy.fftpack import fft, fftfreq, ifft
from scipy.signal import butter, filtfilt 
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pylab import cm
import pywt
import os
import math
import time

mpl.rcParams['font.family'] = 'TeX Gyre Schola Math'
# mpl.rcParams['font.family'] = 'monospace'
plt.rcParams['font.size'] = 40
plt.rcParams['axes.linewidth'] = 2

label_list = [
    ('Idle', 'A', 0, 'r'), 
    ('Stone', 'B', 1, 'g'), 
    ('Tissue', 'C', 2, 'b')
]
interval_num = 50


#data
data, labels = {}, {}
for idx, label in enumerate(label_list):
    data[label[0]] = np.array(pd.read_excel('./Preliminary/data_arranged_dep.xlsx', sheet_name='Sheet1', header=0, usecols=label[1]))
    data[label[0]] = data[label[0]][~np.isnan(data[label[0]])]
    # print(data[label])
    # print(np.shape(data[label])[0])
    labels[label[0]] = np.repeat(label[2], np.shape(data[label[0]])[0])


# idle_data = data['Idle']
# n = np.shape(idle_data)[0]
# T = 0.01
# fs = n/T
# cutoff = 2 
# nyq = 0.5*fs
# order = 2
# def butter_lowpass_filter(data, cutoff, fs, order):
#     normal_cutoff = cutoff/nyq
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     y=filtfilt(b, a, data)
#     return y 

# data['Idle'] = np.ones(n) #butter_lowpass_filter(idle_data, cutoff, fs, order)

data_sliced, label_sliced_1d = {}, {}
data_converted = {}
# print(data.keys())
num_list = []
fft_dict = {}
plt.figure()
for idx1, label in enumerate(label_list):
    data_normalized = data[label[0]]/np.max(data[label[0]])
    num = math.floor(np.shape(data[label[0]])[0]/interval_num)
    num_list.append(num)
    data_sliced[label[0]] = np.reshape(data_normalized[:num*interval_num], (-1,interval_num))
    n = np.shape(data_sliced[label[0]])[1]
    d = 0.01
    freqs = fftfreq(
        n, 
        d = d
    )
    mask = freqs > 0
    nwaves = freqs*n
    m = np.shape(data_sliced[label[0]])[0]
    
    fft_theo = np.zeros((n,))
    for idx2 in range(m):
        fft_vals = fft(data_sliced[label[0]][idx2,:])
        fft_norm = fft_vals*(1.0/n)
        fft_theo_temp = 2.0*abs(fft_norm)
        fft_theo += fft_theo_temp

    plt.subplot(len(label_list), 1, idx1+1)
    plt.bar(
        freqs[mask]*n, 
        fft_theo[mask]/m, 
        width=10,
        label="{0}".format("{0}".format(label[0])),
        color=label[3]
    )
    # plt.title(label[0])
    plt.axhline(y=0, linewidth=1, color='k')
    plt.ylim(3e-4, 3e-2)
    plt.yscale('log')
    plt.xticks(np.around(np.arange(0,(0.5/d)*interval_num,step=100), decimals=-1), rotation=90)
    plt.ylabel('Logscale')
    plt.legend()
    fft_dict[label[0]] = [freqs[mask]*n, fft_theo[mask]/m]
plt.xlabel('Frequency (Hz)')
plt.suptitle('FFT values of each states')
# plt.show()


# len_check = []
# for label in label_list:
#     fft_len = len(fft_dict[label[0]]) 
#     len_check.append(fft_len)
# fft_len = int(max(len_check))
# ind = np.arange(fft_len)
ind = np.arange(len(fft_dict[label_list[0][0]][0]))
width = 0.25 #0.27

lw = 1
fig = plt.figure()
ax = fig.add_subplot(111)
rects_idle = ax.bar(
    ind, 
    fft_dict[label_list[0][0]][1], 
    width, 
    color='#E3E3E3',
    edgecolor='black',
    linewidth=lw,
    hatch='//'
)
rects_stone = ax.bar(
    ind+width, 
    fft_dict[label_list[1][0]][1], 
    width, 
    color='#7C7C7C',
    edgecolor='black',
    linewidth=lw,
    # hatch='..'
)
rects_tissue = ax.bar(
    ind+width*2, 
    fft_dict[label_list[2][0]][1], 
    width, 
    color='#000000',
    edgecolor='black',
    linewidth=lw,
    # hatch='||'
)
# rects_idle = ax.bar(
#     ind, 
#     fft_dict[label_list[0][0]][1], 
#     width, 
#     color='#5C94D6',
#     edgecolor='black',
#     linewidth=lw,
#     # hatch='//'
# )
# rects_stone = ax.bar(
#     ind+width, 
#     fft_dict[label_list[1][0]][1], 
#     width, 
#     color='#4474B8',
#     edgecolor='black',
#     linewidth=lw,
#     hatch='//'
# )
# rects_tissue = ax.bar(
#     ind+width*2, 
#     fft_dict[label_list[2][0]][1], 
#     width, 
#     color='#0350BD',
#     edgecolor='black',
#     linewidth=lw,
#     # hatch='\\'
# )

ax.set_ylabel('Logscale Magnitude')
ax.set_xlabel('Frequency (Hz)')
# ax.grid(axis='y')
ax.set_xticks(ind+width)
xt = fft_dict[label_list[0][0]][0].astype(int)
ax.set_xticklabels(xt, rotation=60)
ax.legend(
    (rects_idle[0], rects_stone[0], rects_tissue[0]), 
    (label_list[0][0], label_list[1][0], label_list[2][0])
)
fig.subplots_adjust(bottom=0.43)
plt.yscale('log')
plt.show()
