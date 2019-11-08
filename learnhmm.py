import sys
import csv
import math as m
import numpy as np
import collections

#Read training data
data_train = []
with open(sys.argv[1], 'r') as f:
    for line in f:
        temp_line = line
        data_train.append(temp_line)
data_train = [i.strip() for i in data_train]

word_index = []
with open(sys.argv[2], 'r') as f:
    for line in f:
        temp_line = line
        word_index.append(temp_line)
word_index = [i.strip() for i in word_index]

tag_index = []
with open(sys.argv[3], 'r') as f:
    for line in f:
        temp_line = line
        tag_index.append(temp_line)
tag_index = [i.strip() for i in tag_index]

split_data = []
word_data = []
tag_ind_data = []
word_ind_data = []
seq_length = len(data_train)
for i in range(seq_length):
    row = data_train[i]
    entity = row.split(' ')
    temp_split = []
    temp_word = []
    temp_t_ind = []
    temp_w_ind = []
    for e in entity:
        word, tag = e.split('_')
        w_ind = word_index.index(word)
        t_ind = tag_index.index(tag)
        temp_split.append((w_ind, t_ind))
        temp_word.append(word)
        temp_w_ind.append(w_ind)
        temp_t_ind.append(t_ind)
    split_data.append(temp_split)
    word_data.append(temp_word)
    tag_ind_data.append(temp_t_ind)
    word_ind_data.append(temp_w_ind)

B = np.zeros((len(tag_index), len(word_index)))
for j, k in zip(word_ind_data, tag_ind_data):
    for word, tag in zip(j, k):
        B[tag][word] += 1
B_den = np.sum((B+1), axis=1)
B_den = B_den.reshape(-1, 1)
B = (B+1)/B_den

PI = np.zeros(len(tag_index))
for i in tag_ind_data:
    PI[i[0]] += 1
PI = (PI+1)/np.sum(PI+1)

A = np.zeros((len(tag_index), len(tag_index)))
tags = []
for row in split_data:
    tags = [i[1] for i in row]
    for a,b in zip(tags[1:], tags):
        A[b][a] += 1
A_den = np.sum((A+1), axis=1)
A_den = A_den.reshape(-1, 1)
A = (A+1)/A_den

hmmprior = PI
hmmemit = B
hmmtrans = A

with open(sys.argv[4], 'wb') as f:
    np.savetxt(sys.argv[4], PI)

with open(sys.argv[5], 'wb') as f:
    np.savetxt(sys.argv[5], B)

with open(sys.argv[6], 'wb') as f:
    np.savetxt(sys.argv[6], A)

# with open("hmmtrans.txt", 'r') as f1:
#      with open("g_hmmtrans.txt", 'r') as f2:
#          for line1, line2 in zip(f1, f2):
#              if line1 == line2:
#                  print("yeap")
#              else:
#                  print("nope")
