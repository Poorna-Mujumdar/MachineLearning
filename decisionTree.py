
import sys
import csv
import math as m
import numpy as np
import collections

#Read training data
with open(sys.argv[1], 'r') as f:
    csv_f = csv.reader(f)
    data = []
    for row in csv_f:
        data.append(row)

#Read test data
with open(sys.argv[2], 'r') as f:
    csv_f = csv.reader(f)
    data_test = []
    for row in csv_f:
        data_test.append(row)

#FOR TRAINING DATA
#Convert dataset to numpy array
data = np.array(data)
#Separate attribute names (header row)
attr_list = data[0]
attr_list = np.delete(attr_list, -1)
#print(attr_list)
#Delete header row from the dataset
data = np.delete(data, 0, 0)

#FOR TEST DATA
#Convert dataset to numpy array
data_test = np.array(data_test)
#Separate attribute names (header row)
attr_list_test = data_test[0]
attr_list_test = np.delete(attr_list_test, -1)
#Delete header row from the dataset
data_test = np.delete(data_test, 0, 0)

#Defining class Node
class Node:
    def __init__(self,
                 data,
                 left,
                 right,
                 attribute,
                 index,
                 value,
                 label):
        self.dataset = data
        self.left = left
        self.right = right
        self.attribute = attribute
        self.index = index
        self.value = value
        self.label = label

def class_counts(dataset):
    y = []
    for row in dataset:
        y.append(row[-1])
    label, counts = np.unique(y, return_counts = True)
    return label, counts

#Entropy calculaion
def entropy(labels, base=None):
    Entropy = 0
    if len(labels) <= 1:
        return 0
    value,counts = np.unique(labels, return_counts=True)
    probs = (counts) / (float(len(labels)))
    Entropy -= (probs*np.log(probs)/np.log(2)).sum()
    return Entropy

#Information gain calculation
def info_gain(y, x):
    weighted_entro = entropy(y)
    value, counts = np.unique(x, return_counts=True)
    probs_attr = (counts)/(float(len(x)))

    for p, v in zip(probs_attr, value):
        weighted_entro -= p * entropy(y[x == v])
    return weighted_entro

#Defining function to print
#def pretty_print(root, n, x):
#    label, counts = class_counts(root.dataset)
#    if len(counts) > 1:

#        if n == 1:
#            print('[{} {} /{} {}]'.format(counts[0], label[0], counts[1], label[1]))

#        else:
#            print('|'*(n), root.attribute, '=', root.value, ':', ('[{} {} /{} {}]'.format(counts[0], label[0], counts[1], label[1])))

#    else:
#        print('|'*(n), root.attribute, '=', root.value, ':', ('[{} {}]'.format(counts[0], label[0])))


#Defining the root
root = Node(data = data, left = None, right = None, attribute = None, index = None, value = None, label = None)
max_depth = sys.argv[3]
max_depth = int(max_depth)

#Defining tree
def tree(root, depth, m_attr_list):
    y = root.dataset[:,-1]
    x = np.delete(root.dataset, -1, 1)
    temp =[]

    # If there could be no split, return the original set
    if len(root.dataset) == 0:
        count_y = collections.Counter(y)
        root.label = max(count_y, key = count_y.get)
        return root

    elif len(x[0]) == 0:
        #print "Attributes over"
        count_y = collections.Counter(y)
        root.label = max(count_y, key = count_y.get)
        return root

    elif depth > max_depth:
        #print "MAX depth reached"
        count_y = collections.Counter(y)
        root.label = max(count_y, key = count_y.get)
        return root

    else:
        # to select attribute with highest information gain

        gain = np.array([info_gain(y, x_attr) for x_attr in x.T])
        best_attr_index = np.argmax(gain)

        best_attribute = m_attr_list[best_attr_index]
        root.index = best_attr_index

        #Calculating this only to check if it's zero
        best_attr = np.amax(gain)
        m_attr_list = np.delete(m_attr_list, best_attr_index)

        #To denote a cell as certain "value" for splitting the dataset
        temp = root.dataset[:,best_attr_index]
        temp = np.asarray(temp)
        value, counts = np.unique(temp, return_counts = True)

        #Remove this while printing tree
        root.value = value

        #Start printing tree
        #pretty_print(root, depth, x)

        #If there's no gain, return root
        if best_attr <= 0:
            count_y = collections.Counter(y)
            root.label = max(count_y, key = count_y.get)
            return root

        #Split to left child & right child
        else:
            left_child, right_child = [],[]
            for row in root.dataset:
                if row[best_attr_index] == value[0]:
                    left_child.append(row)

                else:
                    right_child.append(row)

            #convert left and right into numpy arrays
            left_child = np.asarray(left_child)
            right_child = np.asarray(right_child)

            if len(value) == 1:
                left_child = np.delete(left_child, best_attr_index, 1)
                left = Node(data = left_child, left = None, right = None, attribute = best_attribute, index = best_attr_index, value = value[0], label = None)

            else:
                left_child = np.delete(left_child, best_attr_index, 1)
                left = Node(data = left_child, left = None, right = None, attribute = best_attribute, index = best_attr_index, value = value, label = None)

                right_child = np.delete(right_child, best_attr_index, 1)
                right = Node(data = right_child, left = None, right = None, attribute = best_attribute, index = best_attr_index, value = value, label = None)

                #recurse
                root.left = tree(left, depth+1, m_attr_list)
                root.right = tree(right, depth+1, m_attr_list)

                return root


tree(root, 1, attr_list)

def search_tree(row, tree_node):
    #if tree_node != None:
    if (tree_node.label != None):
        return tree_node.label

    feature = row[tree_node.index]
    #print("Feature:", feature)
    if tree_node.value[0] == feature:
        #print("GOING LEFT")
        row = np.delete(row, tree_node.index)
        label = search_tree(row, tree_node.left)

    else:
        #print("GOING RIGHT")
        row = np.delete(row, tree_node.index)
        label = search_tree(row, tree_node.right)
    return label

labels_train = []
for row in root.dataset:
    labels_train.append(search_tree(row, root))

#To print training data labels
with open(sys.argv[4], 'w') as f:
    for item in labels_train:
        f.write("%s\n" % item)

labels_test = []
for row in data_test:
    labels_test.append(search_tree(row, root))

#To print test data labels
with open(sys.argv[5], 'w') as f:
    for item in labels_test:
        f.write("%s\n" % item)

#TO DETERMINE ERROR RATE
#Training error
train_error = 1 - np.mean(np.asarray(labels_train) == np.asarray(data)[:,-1])
train_error = np.asscalar(train_error)

labels = []
for row in data_test:
    labels.append(search_tree(row, root))

#Testing error
test_error = 1 - np.mean(np.asarray(labels_test) == np.asarray(data_test)[:,-1])
test_error = np.asscalar(test_error)

c1 = str(train_error)
c2 = str(test_error)

#To write the metrix file output
with open(sys.argv[6], 'w') as file:
    line1 = "error(train): " + c1
    line2 = "error(test): " + c2
    file.write("%s\n%s\n" % (line1, line2))
