import sys
import csv
import math as m
import numpy as np
import collections

#Read dictionary file
dict_input = sys.argv[4]

#Read training file, create list for lables, seperate fetures
with open(sys.argv[1], 'r') as f:
    Y_train=[]
    X_train=[]
    for line in f:
        Yi = float(line[0])
        Y_train.append(Yi)

        line_list = line.strip()
        line_list = line_list.split("\t")
        temp_words = []
        Xi = line_list[1:]

        for word in Xi:
            new_word = word[:-2]
            temp_words.append(new_word)
        X_train.append(temp_words)

#Read validation file, create list for lables, seperate fetures
with open(sys.argv[2], 'r') as f:
    Y_valid =[]
    X_valid =[]
    for line in f:
        Yi = float(line[0])
        Y_valid.append(Yi)

        line_list = line.strip()
        line_list = line_list.split("\t")
        temp_words = []
        Xi = line_list[1:]

        for word in Xi:
            new_word = word[:-2]
            temp_words.append(new_word)
        X_valid.append(temp_words)

#Read test file, create list for lables, seperate fetures
with open(sys.argv[3], 'r') as f:
    Y_test =[]
    X_test =[]
    for line in f:
        Yi = float(line[0])
        Y_test.append(Yi)

        line_list = line.strip()
        line_list = line_list.split("\t")
        temp_words = []
        Xi = line_list[1:]

        for word in Xi:
            new_word = word[:-2]
            temp_words.append(new_word)
        X_test.append(temp_words)

#Add bias column
for row in X_train:
    row.append(-1)

#Convert X to a list of dicts
dict_X = []
for row in X_train:
    temp_dict = {}
    for words in row:
        temp_dict[words] = 1
    dict_X.append(temp_dict)

#Convert X_test  to a list of dicts
dict_X_test = []
for row in X_test:
    temp_dict = {}
    for words in row:
        temp_dict[words] = 1
    dict_X_test.append(temp_dict)

theta = {}
l_rate = 0.1

#Dot product function
def dot_product(dict1, dict2):
    sum = 0
    for k in dict1:
        if k in dict2:
            sum += dict1[k]*dict2[k]
    return sum

#Reading num epoch value
num_epoch = int(sys.argv[8])

#Stochastic gradient descent function
def s_grad_d(X, Y, epoch):
    while epoch < num_epoch:
        for x_i,y_i in zip(X, Y):
            dot_prod = dot_product(theta, x_i)
            expo = m.exp(dot_prod)
            error = y_i - (expo/(1+expo))

            for weight_j in x_i:
                if weight_j  in theta:
                    theta[weight_j] += (l_rate*error)
                else:
                    theta[weight_j] = l_rate*error
        epoch = epoch + 1

    return theta

#Running SGD function on test dataset
s_grad_d(dict_X, Y_train, 0)

#To generate labels
def search_model(X, theta):
    label_out = []
    for row in X:
        if dot_product(theta, row) >= 0:
            label = 1
        else:
            label = 0
        label_out.append(label)
    return label_out

#Training labels and training erroe
labels_train = []
labels_train = search_model(dict_X,theta)
train_error = 1 - np.mean(np.asarray(labels_train) == np.asarray(Y_train))
train_error = np.asscalar(train_error)

#To print test data labels
with open(sys.argv[5], 'w') as f:
    for item in labels_train:
        f.write("%s\n" % item)

#test labels and test error
labels_test = []
labels_test = search_model(dict_X_test,theta)
test_error = 1 - np.mean(np.asarray(labels_test) == np.asarray(Y_test))
test_error = np.asscalar(test_error)

#To print test data labels
with open(sys.argv[6], 'w') as f:
    for item in labels_test:
        f.write("%s\n" % item)

#To write the metrix file output
c1 = str(train_error)
c2 = str(test_error)

with open(sys.argv[7], 'w') as file:
    line1 = "error(train): " + c1
    line2 = "error(test): " + c2
    file.write("%s\n%s\n" % (line1, line2))
