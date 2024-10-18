# build a multi class decision tree classifier using the training data without using any libraries
# and test it using the test data


import numpy as np
import pandas as pd
import math
import random
import sys
import os
import cv2
import pickle



# Reading the Data

# read the face images from the data/train directory
train_images_person = []

for filename in os.listdir("/content/drive/MyDrive/data/train/person"):
    # read the image
    # print("/content/drive/MyDrive/data/train/person/" + filename)
    
    
    image = cv2.imread("/content/drive/MyDrive/data/train/person/" + filename, 1)
    if image is None:
      continue

    # print(image.shape)

    # resize the image
    # image = cv2.resize(image, (28, 28))

    # flatten the image
    image = image.flatten()

    # add the image to the list of images
    train_images_person.append(image)

# read the car images from the data/train directory
train_images_car = []

for filename in os.listdir("/content/drive/MyDrive/data/train/car"):
    # read the image
    # print("/content/drive/MyDrive/data/train/car/" + filename)
    
    
    image = cv2.imread("/content/drive/MyDrive/data/train/car/" + filename, 1)
    if image is None:
      continue

    # print(image.shape)

    # resize the image
    # image = cv2.resize(image, (28, 28))

    # flatten the image
    image = image.flatten()

    # add the image to the list of images
    train_images_car.append(image)

# read the airplane images from the data/train directory
train_images_airplane = []

for filename in os.listdir("/content/drive/MyDrive/data/train/airplane"):
    # read the image
    # print("/content/drive/MyDrive/data/train/airplane/" + filename)
    
    
    image = cv2.imread("/content/drive/MyDrive/data/train/airplane/" + filename, 1)
    if image is None:
      continue

    # print(image.shape)

    # resize the image
    # image = cv2.resize(image, (28, 28))

    # flatten the image
    image = image.flatten()

    # add the image to the list of images
    train_images_airplane.append(image)


# read the dog images from the data/train directory
train_images_dog = []

for filename in os.listdir("/content/drive/MyDrive/data/train/dog"):
    # read the image
    # print("/content/drive/MyDrive/data/train/dog/" + filename)
    
    
    image = cv2.imread("/content/drive/MyDrive/data/train/dog/" + filename, 1)
    if image is None:
      continue

    # print(image.shape)

    # resize the image
    # image = cv2.resize(image, (28, 28))

    # flatten the image
    image = image.flatten()

    # add the image to the list of images
    train_images_dog.append(image)


#Label the classes as given : Cars - 0, Faces - 1, Airplanes - 2, Dogs - 3.


# create the training data
train_data = train_images_person + train_images_car + train_images_airplane + train_images_dog

# create the training labels
train_labels = [1] * len(train_images_person) + [0] * len(train_images_car) + [2] * len(train_images_airplane) + [3] * len(train_images_dog)

# Build the decision tree classifier using the training data and labels


# calculate the entropy of the given data
def calculate_entropy(data):
    # calculate the frequency of each label
    frequency = {}
    for label in data:
        if label not in frequency:
            frequency[label] = 0
        frequency[label] += 1

    # calculate the entropy
    entropy = 0
    for label in frequency:
        probability = frequency[label] / len(data)
        entropy += probability * math.log(probability, 2)

    return -entropy


# calculate the information gain of the given data
def calculate_information_gain(data, split_data):
    # calculate the total entropy
    total_entropy = calculate_entropy(data)

    # calculate the weighted entropy
    weighted_entropy = 0
    for subset in split_data:
        weighted_entropy += (len(subset) / len(data)) * calculate_entropy(subset)

    # calculate the information gain
    information_gain = total_entropy - weighted_entropy

    return information_gain


# split the data based on the given feature
def split_data(data, labels, feature):
    # create the split data
    split_data = {}
    for i in range(len(data)):
        if data[i][feature] not in split_data:
            split_data[data[i][feature]] = []
        split_data[data[i][feature]].append(labels[i])

    return split_data


# find the best feature to split the data
def find_best_feature(data, labels):
    # calculate the initial information gain
    best_feature = None
    best_information_gain = 0

    # calculate the information gain for each feature
    for feature in range(len(data[0])):
        split_data = split_data(data, labels, feature)
        information_gain = calculate_information_gain(labels, split_data.values())

        # update the best feature and information gain
        if information_gain > best_information_gain:
            best_feature = feature
            best_information_gain = information_gain

    return best_feature


# find the most common label in the data
def find_most_common_label(labels):
    # calculate the frequency of each label
    frequency = {}
    for label in labels:
        if label not in frequency:
            frequency[label] = 0
        frequency[label] += 1

    # find the most common label
    most_common_label = None
    for label in frequency:
        if most_common_label is None or frequency[label] > frequency[most_common_label]:
            most_common_label = label

    return most_common_label


# build the decision tree
def build_decision_tree(data, labels):
    # create the decision tree
    decision_tree = {}

    # find the most common label in the data
    most_common_label = find_most_common_label(labels)

    # if the data is empty or the labels are the same, return the most common label
    if len(data) == 0 or labels.count(labels[0]) == len(labels):
        return most_common_label

    # if there is only one feature, return the most common label
    if len(data[0]) == 1:
        return most_common_label

    # find the best feature to split the data
    best_feature = find_best_feature(data, labels)

    # split the data based on the best feature
    split_data = split_data(data, labels, best_feature)

    # create a subtree for each split
    for feature_value, subset_labels in split_data.items():
        
        # create a subset of the data
        subset = []
        for i in range(len(data)):
            if data[i][best_feature] == feature_value:
                subset.append(data[i])
        

        
        # create a subtree
        

        subtree = build_decision_tree(subset, subset_labels)

        # insert the subtree into the decision tree
        decision_tree[(best_feature, feature_value)] = subtree

    return decision_tree


# build the decision tree
decision_tree = build_decision_tree(train_data, train_labels)

# print the decision tree
print(decision_tree)

# predict the label of the given data
def predict_label(data, decision_tree):
    # if the node is a leaf node, return the label
    if type(decision_tree) is not dict:
        return decision_tree

    # find the best feature to split the data
    feature, value = list(decision_tree.keys())[0]

    # find the subtree
    subtree = None
    if data[feature] == value:
        subtree = decision_tree[(feature, value)]
    else:
        subtree = decision_tree[(feature, not value)]

    # make a prediction
    return predict_label(data, subtree)


# predict the labels of the test data
test_labels = []
for data in test_data:
    label = predict_label(data, decision_tree)
    test_labels.append(label)

# calculate the accuracy of the decision tree
correct = 0
for i in range(len(test_labels)):
    if test_labels[i] == test_labels[i]:
        correct += 1
accuracy = correct / len(test_labels)
print("Accuracy:", accuracy)

# save the decision tree
with open("decision_tree.pickle", "wb") as f:
    pickle.dump(decision_tree, f)

# load the decision tree
with open("decision_tree.pickle", "rb") as f:
    decision_tree = pickle.load(f)










