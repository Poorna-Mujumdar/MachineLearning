import sys
import csv
import math as m
import numpy as np
import collections

#Read test data
data_test = []
with open(sys.argv[1], 'r') as f:
    for line in f:
        temp_line = line
        data_test.append(temp_line)
data_test = [i.strip() for i in data_test]

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

hmmprior = sys.argv[4]
hmmprior = np.loadtxt(hmmprior)

hmmemit = sys.argv[5]
hmmemit = np.loadtxt(hmmemit)

hmmtrans = sys.argv[6]
hmmtrans = np.loadtxt(hmmtrans)

def sort_data(dataset, word_index, tag_index):
    split_data = []
    word_data = []
    tag_data = []
    seq_length = len(dataset)
    for i in range(seq_length):
        row = dataset[i]
        entity = row.split(' ')
        temp_split = []
        temp_word = []
        temp_tag = []
        for e in entity:
            word, tag = e.split('_')
            w_ind = word_index.index(word)
            t_ind = tag_index.index(tag)
            temp_split.append((w_ind, t_ind))
            temp_word.append(word)
            temp_tag.append(tag)
        split_data.append(temp_split)
        word_data.append(temp_word)
        tag_data.append(temp_tag)
    return split_data, word_data, tag_data

split_data_test, word_data_test, tag_data_test = sort_data(data_test, word_index, tag_index)

def forward_algo(row, tag_index, word_index, hmmprior, hmmemit, hmmtrans):
    alpha_n = len(tag_index)
    alpha_m = len(row)
    alpha = np.zeros(shape=(alpha_m, alpha_n))
    for i in range(alpha_m):
        word = row[i]
        w_ind = word_index.index(word)
        if i == 0:
            alpha[i] = hmmprior*(hmmemit[:,w_ind].T)
        else:
            alpha[i] = (hmmemit[:,w_ind].T)*(np.dot(alpha[i-1], hmmtrans))
    return alpha

def backward_algo(row, tag_index, word_index, hmmprior, hmmemit, hmmtrans):
    beta_m = len(row)
    beta_n = len(tag_index)
    beta = np.zeros(shape=(beta_m, beta_n))
    for i in range(beta_m-1, -1, -1):
        word = row[i]
        w_ind = word_index.index(word)
        if i == (beta_m-1):
            beta[i] = np.ones(shape=(1, beta_n))
        else:
            word = row[i+1]
            w_ind = word_index.index(word)
            beta_temp = (hmmemit[:,w_ind].T)*beta[i+1]
            beta[i] = np.dot(beta_temp, hmmtrans.T)
    return beta

def predict(row1, row2, tag_index, word_index, hmmprior, hmmemit, hmmtrans):
    length = len(row1)
    count = 0
    correct_count = 0
    predicted_tag = []
    word_list =[]
    alpha = forward_algo(row1, tag_index, word_index, hmmprior, hmmemit, hmmtrans)
    beta = backward_algo(row1, tag_index, word_index, hmmprior, hmmemit, hmmtrans)
    for t in range(length):
        alpha_t = alpha[t,:]
        beta_t = beta[t,:]
        y_t = np.argmax(np.multiply(alpha_t, beta_t))
        pred_tag = tag_index[y_t]
        actual_tag = row2[t]
        if pred_tag == actual_tag:
            correct_count += 1
        word = row1[t]
        word_list.append(word)
        predicted_tag.append(pred_tag)
        count += 1
    return predicted_tag, word_list, correct_count, count

prediction = []
final_correct_count = 0
final_count = 0
predicted_string = ""
for row1, row2 in zip(word_data_test, tag_data_test):
    pred_row, word, correct_count, count = predict(row1, row2, tag_index, word_index, hmmprior, hmmemit, hmmtrans)
    for (p, w) in zip(pred_row, word):
        temp_string = w + "_" + p + " "
        predicted_string += temp_string
    predicted_string = predicted_string.rstrip()
    predicted_string += "\n"
    final_correct_count = final_correct_count + correct_count
    final_count = final_count + count

def log_likeliehood(row, tag_index, word_index, hmmprior, hmmemit, hmmtrans):
    length = len(row)
    alpha = forward_algo(row, tag_index, word_index, hmmprior, hmmemit, hmmtrans)
    ll = np.log(np.sum(alpha[length-1]))
    return ll

final_ll = []
for row in word_data_test:
    ll = log_likeliehood(row, tag_index, word_index, hmmprior, hmmemit, hmmtrans)
    final_ll.append(ll)

avg_ll = sum(final_ll)/len(final_ll)
accuracy = float(final_correct_count)/float(final_count)

#To write predicted output file
with open(sys.argv[7], 'w') as f:
    f.writelines(predicted_string)

#To write the metrics file output
c1 = str(avg_ll)
c2 = str(accuracy)
with open(sys.argv[8], 'w') as file:
    line1 = "Average Log-Likelihood: " + c1
    line2 = "Accuracy: " + c2
    file.write("%s\n%s\n" % (line1, line2))

# with open(sys.argv[7], 'r') as f1:
#       with open("predictedtest.txt", 'r') as f2:
#           for line1, line2 in zip(f1, f2):
#               if line1 == line2:
#                   print("yeap")
#               else:
#                   print("nope")
