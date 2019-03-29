import sys
import csv
import math as m
import numpy as np
import collections

#Read training data
with open(sys.argv[1], 'r') as f:
    csv_f = csv.reader(f)
    data_train = []
    for row in csv_f:
        data_train.append(row)

#Read test data
with open(sys.argv[2], 'r') as f:
    csv_f = csv.reader(f)
    data_test = []
    for row in csv_f:
        data_test.append(row)

#FOR TRAINING DATA
#Convert dataset to numpy array
data_train = np.array(data_train, dtype = int)

#Separate labels & features
Y_train = data_train[:, 0]
X_train = np.delete(data_train, 0, 1)
Y_unique_train = np.unique(Y_train)

#FOR TEST DATA
#Convert dataset to numpy array
data_test = np.array(data_test, dtype = int)

#Separate labels & features
Y_test = data_test[:, 0]
X_test = np.delete(data_test, 0, 1)
Y_unique_test = np.unique(Y_test)

#Hidden layers
hid_units = sys.argv[7]
D = int(hid_units)
#Random or zeros
init_flag = int(sys.argv[8])
#Epoch
num_epoch = int(sys.argv[6])
#Learning rate
lr = float(sys.argv[9])
#No of features
num_features = np.size(X_train, 1)
M = int(num_features)
#No. of unique labels
num_output = np.size(Y_unique_train)
K = int(num_output)

#Add bias vector
X_train = np.insert(X_train, 0, values=1, axis=1)
X_test = np.insert(X_test, 0, values=1, axis=1)

#One hot coding of labels
def one_hot_coded(val):
    one_hot_y = [0]*K
    one_hot_y[val] = 1
    one_hot_y = np.asarray(one_hot_y, dtype=np.float32)
    return one_hot_y

#Initialize alpha and beta
if init_flag == 1:
    alpha_star = np.random.uniform(low=-0.1, high=0.1, size=(D,M))
    beta_star = np.random.uniform(low=-0.1, high=0.1, size=(K,D))

elif init_flag == 2:
    alpha_star = np.zeros(shape=(D,M))
    beta_star = np.zeros(shape=(K,D))

alpha = np.insert(alpha_star, 0, values=0, axis=1)
beta = np.insert(beta_star, 0, values=0, axis=1)

def linear_forward(val, weight):
    return np.matmul(weight, val)

def sigmoid_forward(val):
    return 1/(1+np.exp(-val))

def softmax_forward(val):
    return np.divide(np.exp(val),np.sum(np.exp(val)))

def cross_entropy_forward(val, val_hat):
    return -np.matmul(val.T,np.log(val_hat))

def cross_entropy_backward(val, val_hat):
    return -np.divide(val,val_hat)

#Forward propagation funtion
def forward_prop(x, y, alpha, beta):
    a = linear_forward(x, alpha)
    z = sigmoid_forward(a)
    z = np.append(1, z)
    b = linear_forward(z, beta)
    y_hat = softmax_forward(b)
    J = cross_entropy_forward(y, y_hat)
    return x, a, z, b, y_hat, J

def softmax_backward(val, val_hat):
    return val_hat - val

def linear_backward(val, weight, grad):
    if (len(val.shape) == 1):
        val = val.reshape(-1, 1)
    if (len(grad.shape) == 1):
        grad = grad.reshape(1, -1)
    op_1 = np.matmul(grad.T, val.T)
    op_2 = np.matmul(weight.T, grad.T)
    op_2 = op_2[1:]
    return op_1, op_2

def linear_backward_2(val, weight, grad):
    if (len(val.shape) == 1):
        val = val.reshape(-1, 1)
    if (len(grad.shape) == 1):
        grad = grad.reshape(1, -1)
    op_1 = np.matmul(grad.T, val.T)
    return op_1

def sigmoid_backward(val, grad):
    zz = np.multiply(val, 1-val)
    return np.multiply(grad.T, zz)

#Backward propagation function
def back_prop(x, y, alpha, beta, inter):
    x, a, z, b, y_hat, J = inter
    gj = 1
    gy = cross_entropy_backward(y, y_hat)
    gb = softmax_backward(y, y_hat)
    gbeta, gz = linear_backward(z, beta, gb)
    ga = sigmoid_backward(z[1:], gz)
    galpha = linear_backward_2(x, alpha, ga)
    return galpha, gbeta

metrics = open(sys.argv[5], 'w')
mtc=0
mtestc=0
for i in range(num_epoch):
    for x, y_i in zip(X_train, Y_train):
        y = one_hot_coded(y_i)
        x, a, z, b, y_hat, J = forward_prop(x, y, alpha, beta)
        g_alpha, g_beta = back_prop(x, y, alpha, beta, [x, a, z, b, y_hat, J])
        beta = beta - (np.multiply(lr, g_beta))
        alpha = alpha - (np.multiply(lr, g_alpha))

    train_cross_entropy = []
    for x, y_i in zip(X_train, Y_train):
        y = one_hot_coded(y_i)
        x, a, z, b, y_hat, J = forward_prop(x, y, alpha, beta)
        train_cross_entropy.append(J)

    test_cross_entropy = []
    for x, y_i in zip(X_test, Y_test):
        y = one_hot_coded(y_i)
        x, a, z, b, y_hat, J = forward_prop(x,y, alpha, beta)
        test_cross_entropy.append(J)

    mtc = sum(train_cross_entropy)/len(train_cross_entropy)
    mtestc = sum(test_cross_entropy)/len(test_cross_entropy)
    metrics.write("epoch="+str(i+1)+" crossentropy(train) : "+ str(mtc)+"\n")
    metrics.write("epoch="+str(i+1)+" crossentropy(test) : "+ str(mtestc)+"\n")

def predict_labels(x, y, alpha, beta):
    y_predict = []
    for x, y_i in zip(x, y):
        y = one_hot_coded(y_i)
        x, a, z, b, y_hat, J = forward_prop(x, y, alpha, beta)
        y_predict.append(np.argmax(y_hat))
    return y_predict

labels_train = predict_labels(X_train, Y_train, alpha, beta)

#To print training data labels
with open(sys.argv[3], 'w') as f:
    for item in labels_train:
        f.write("%s\n" % item)

labels_test = predict_labels(X_test, Y_test, alpha, beta)

#To print training data labels
with open(sys.argv[4], 'w') as f:
    for item in labels_test:
        f.write("%s\n" % item)

#TO DETERMINE ERROR RATE
#Trainin00g error
train_error = 1 - np.mean(np.asarray(labels_train) == np.asarray(data_train)[:,0])
train_error = np.asscalar(train_error)

#Test error
test_error = 1 - np.mean(np.asarray(labels_test) == np.asarray(data_test)[:,0])
test_error = np.asscalar(test_error)

#To write the error to metrics file output
metrics.write("error(train):"+str(train_error)+"\n")
metrics.write("error(test):"+str(test_error)+"\n")
