# build a decision tree classifier using the training data without using any libraries
# and test it using the test data

import numpy as np
import pandas as pd
import math
import random
import sys
import os
import cv2
import pickle
import time



# Reading the Data


# read the face images from the data/train directory
train_images_face = []

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
    train_images_face.append(image)

# read the rest of images from the data/train directory
train_images_rest = []
for filename in os.listdir("/content/drive/MyDrive/data/train/airplane"):
    # read the image
    image = cv2.imread("/content/drive/MyDrive/data/train/airplane/" + filename, 1)

    if image is None:
      continue

    # resize the image
    # image = cv2.resize(image, (28, 28))

    # flatten the image
    image = image.flatten()

    # add the image to the list of images
    train_images_rest.append(image)

for filename in os.listdir("/content/drive/MyDrive/data/train/car"):
    # read the image
    image = cv2.imread("/content/drive/MyDrive/data/train/car/" + filename, 1)

    if image is None:
      continue

    # resize the image
    # image = cv2.resize(image, (28, 28))

    # flatten the image
    image = image.flatten()

    # add the image to the list of images
    train_images_rest.append(image)

for filename in os.listdir("/content/drive/MyDrive/data/train/dog"):
    # read the image
    image = cv2.imread("/content/drive/MyDrive/data/train/dog/" + filename, 1)

    if image is None:
      continue

    # resize the image
    # image = cv2.resize(image, (28, 28))

    # flatten the image
    image = image.flatten()

    # add the image to the list of images
    train_images_rest.append(image)



# train the decision tree
# create the training data from the face images
train_data_face = []
for image in train_images_face:
    #convert the image to a list
    # image = image.flatten().tolist()
    # print(image.shape)

    # add the image to the training data
    train_data_face.append([image, 1])

# create the training data from the rest of images
train_data_rest = []
for image in train_images_rest:
    #convert the image to a list
    # image = image.flatten().tolist()
    
    # add the image to the training data
    train_data_rest.append([image, 0])  

# create the training data
train_data = train_data_face + train_data_rest

# Extract all images and label all face images as 1 and rest as 0
train_data = []
for image in train_images_face:
    train_data.append([image, 1])
for image in train_images_rest:
    train_data.append([image, 0])



# Build the decision tree

# calculate the entropy of the training data
def entropy(data):
    # calculate the number of face images and the number of rest of images
    face_images = 0
    rest_images = 0
    for image in data:
        if image[1] == 1:
            face_images += 1
        else:
            rest_images += 1

    # calculate the entropy
    entropy = 0
    if face_images > 0:
        entropy -= (face_images / len(data)) * math.log2(face_images / len(data))
    if rest_images > 0:
        entropy -= (rest_images / len(data)) * math.log2(rest_images / len(data))

    return entropy

# calculate the information gain of the training data
def information_gain(data, feature):
    # calculate the entropy of the training data
    entropy_data = entropy(data)

    # calculate the entropy of the training data after splitting on the feature
    entropy_feature = 0
    for value in range(0, 256):
        # split the data
        split_data = []
        for image in data:
            # print(image[0].shape)
            if image[0][feature] == value:
                split_data.append(image)

        # calculate the entropy of the split data
        entropy_split_data = entropy(split_data)

        # calculate the entropy of the feature
        entropy_feature += (len(split_data) / len(data)) * entropy_split_data

    # calculate the information gain
    information_gain = entropy_data - entropy_feature

    return information_gain

# build the decision tree using max_depth and min_samples_split
def build_decision_tree(data, max_depth, min_samples_split):
    # if the data is empty or the max_depth is 0 or the number of samples is less than min_samples_split
    # then return the most common class
    if len(data) == 0 or max_depth == 0 or len(data) < min_samples_split:
        # count the number of face images and the number of rest of images
        face_images = 0
        rest_images = 0
        for image in data:
            if image[1] == 1:
                face_images += 1
            else:
                rest_images += 1

        # return the most common class
        if face_images > rest_images:
            return 1
        else:
            return 0

    # calculate the information gain of the data for each feature
    information_gains = []
    for feature in range(0, 3072):
        information_gains.append(information_gain(data, feature))

    # find the feature with the maximum information gain
    feature = np.argmax(information_gains)

    # split the data on the feature with the maximum information gain
    split_data = []
    for value in range(0, 256):
        # create the split data
        split_data.append([])
        for image in data:
            if image[0][feature] == value:
                split_data[value].append(image)

    # create the decision tree
    decision_tree = {}
    for value in range(0, 256):
        # build the decision tree recursively
        decision_tree[value] = build_decision_tree(split_data[value], max_depth - 1, min_samples_split)

    return [feature, decision_tree]






# Validation data

#read the validation images

# read the face images from the data/validation directory
val_images_face = []

for filename in os.listdir("/content/drive/MyDrive/data/validation/person"):
    # read the image
    # print("/content/drive/MyDrive/data/train/person/" + filename)
    
    
    image = cv2.imread("/content/drive/MyDrive/data/validation/person/" + filename, 1)
    if image is None:
      continue

    # print(image.shape)

    # resize the image
    # image = cv2.resize(image, (28, 28))

    # flatten the image
    image = image.flatten()

    # add the image to the list of images
    val_images_face.append(image)

# read the non-face images from the data/validation directory

val_images_rest = []
for filename in os.listdir("/content/drive/MyDrive/data/validation/airplane"):
    # read the image
    image = cv2.imread("/content/drive/MyDrive/data/validation/airplane/" + filename, 1)

    if image is None:
      continue

    # resize the image
    # image = cv2.resize(image, (28, 28))

    # flatten the image
    image = image.flatten()

    # add the image to the list of images
    val_images_rest.append(image)

for filename in os.listdir("/content/drive/MyDrive/data/validation/car"):
    # read the image
    image = cv2.imread("/content/drive/MyDrive/data/validation/car/" + filename, 1)

    if image is None:
      continue

    # resize the image
    # image = cv2.resize(image, (28, 28))

    # flatten the image
    image = image.flatten()

    # add the image to the list of images
    val_images_rest.append(image)

for filename in os.listdir("/content/drive/MyDrive/data/validation/dog"):
    # read the image
    image = cv2.imread("/content/drive/MyDrive/data/validation/dog/" + filename, 1)

    if image is None:
      continue

    # resize the image
    # image = cv2.resize(image, (28, 28))

    # flatten the image
    image = image.flatten()

    # add the image to the list of images
    val_images_rest.append(image)


# create the validation data    
val_data = []
for image in val_images_face:
    val_data.append([image, 1])

for image in val_images_rest:
    val_data.append([image, 0])







# Classification functions

# classify the validation data and based on the predictions calculate the accuracy, precision and recall

# classify an image
def classify_image(image, decision_tree):
    # if the decision tree is a class then return the class
    if type(decision_tree) is int:
        return decision_tree

    # if the decision tree is a dictionary then classify the image
    else:
        # get the feature and the value of the feature
        feature = decision_tree[0]
        value = image[feature]

        # get the subtree
        subtree = decision_tree[1][value]

        # classify the image
        return classify_image(image, subtree)

# classify the validation data
def classify(data, decision_tree):
    # # if the data is empty then return the most common class
    # if len(data) == 0:
    #     # count the number of face images and the number of rest of images
    #     face_images = 0
    #     rest_images = 0
    #     for image in data:
    #         if image[1] == 1:
    #             face_images += 1
    #         else:
    #             rest_images += 1

    #     # return the most common class
    #     if face_images > rest_images:
    #         return 1
    #     else:
    #         return 0

    # # if the data is not empty then classify the data
    # else:
    #     # classify the data
    #     predictions = []
    #     for image in data:
    #         # classify the image
    #         prediction = classify_image(image[0], decision_tree)

    #         # add the prediction to the list of predictions
    #         predictions.append(prediction)

    #     return predictions
    predictions = []

    for image in data:
        # classify the image
        prediction = classify_image(image[0], decision_tree)

        # add the prediction to the list of predictions
        predictions.append(prediction)
    
    return predictions

    

  


# calculate the accuracy, precision and recall



def accuracy(data, predictions):
    # count the number of correct predictions
    correct_predictions = 0
    for i in range(0, len(data)):
        if data[i][1] == predictions[i]:
            correct_predictions += 1

    # return the accuracy
    return correct_predictions / len(data)

def precision(data, predictions):   
    # count the number of true positives and the number of false positives
    true_positives = 0
    false_positives = 0
    for i in range(0, len(data)):
        if data[i][1] == 1 and predictions[i] == 1:
            true_positives += 1
        elif data[i][1] == 0 and predictions[i] == 1:
            false_positives += 1

    # return the precision
    return true_positives / (true_positives + false_positives)

def recall(data, predictions):
    # count the number of true positives and the number of false negatives
    true_positives = 0
    false_negatives = 0
    for i in range(0, len(data)):
        if data[i][1] == 1 and predictions[i] == 1:
            true_positives += 1
        elif data[i][1] == 1 and predictions[i] == 0:
            false_negatives += 1

    # return the recall
    return true_positives / (true_positives + false_negatives)

    

# calculate the confusion matrix
def confusion_matrix(data, predictions):
    # count the number of true positives, false positives, true negatives and false negatives
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    for i in range(0, len(data)):
        if data[i][1] == 1 and predictions[i] == 1:
            true_positives += 1
        elif data[i][1] == 0 and predictions[i] == 1:
            false_positives += 1
        elif data[i][1] == 0 and predictions[i] == 0:
            true_negatives += 1
        elif data[i][1] == 1 and predictions[i] == 0:
            false_negatives += 1

    # return the confusion matrix
    return [[true_positives, false_positives], [false_negatives, true_negatives]]




# Train the model


# strat time
start_time = time.time()

# build the decision tree
decision_tree = build_decision_tree(train_data, 10, 7)

# end time
end_time = time.time()

# print the time
print("Time to build the information gain decision tree: " + str(end_time - start_time))






print("Information Gain Decision Tree:")



# print the traininig accuracy, precision and recall of the decision tree
predictions = classify(train_data, decision_tree)

print("Training Accuracy: " + str(accuracy(train_data, predictions)))
print("Training Precision: " + str(precision(train_data, predictions)))
print("Training Recall: " + str(recall(train_data, predictions)))

# print the training confusion matrix
print("Training Confusion Matrix:")
print(confusion_matrix(train_data, predictions))





# print the validation accuracy, precision and recall of the decision tree

predictions = classify(val_data, decision_tree)

print("Validation Accuracy: " + str(accuracy(val_data, predictions)))
print("Validation Precision: " + str(precision(val_data, predictions)))
print("Validation Recall: " + str(recall(val_data, predictions)))

# print the validation confusion matrix
print("Validation Confusion Matrix:")
print(confusion_matrix(val_data, predictions))



# save the model to be used later
with open('decision_tree_information_gain.pkl', 'wb') as f:
    pickle.dump(decision_tree, f)

# # load the model
# with open('decision_tree_information_gain.pkl', 'rb') as f:
#     decision_tree = pickle.load(f)


















# Decision Tree using Gini index

# calculate the gini index
def calc_gini_index(data):
    if len(data) == 0:
        return 0
    
    # count the number of face images and the number of rest of images
    face_images = 0
    rest_images = 0
    for image in data:
        if image[1] == 1:
            face_images += 1
        else:
            rest_images += 1

    # calculate the gini index
    gini_index = 1 - (face_images / len(data)) ** 2 - (rest_images / len(data)) ** 2

    return gini_index

# calculate the gini index of the split
def gini_index_split(data, feature, value):
    # split the data
    left_data = []
    right_data = []
    for image in data:
        if image[0][feature] < value:
            left_data.append(image)
        else:
            right_data.append(image)

    # calculate the gini index of the split
    gini_index = (len(left_data) / len(data)) * calc_gini_index(left_data) + (len(right_data) / len(data)) * calc_gini_index(right_data)

    return gini_index

# find the best split
def best_split(data):
    # set the best gini index to a large number
    best_gini_index = 1

    # set the best feature and value to None
    best_feature = None

    best_value = None

    if len(data) == 0:
        return best_feature, best_value

    # loop through all the features
    for feature in range(0, len(data[0][0])):
        # loop through all the values
        for value in range(0, 256):
            # calculate the gini index of the split
            gini_index = gini_index_split(data, feature, value)

            # if the gini index is less than the best gini index then update the best gini index, feature and value
            if gini_index < best_gini_index:
                best_gini_index = gini_index
                best_feature = feature
                best_value = value

    # return the best feature and value
    return best_feature, best_value

# create a leaf node
def create_leaf(data):
    # count the number of face images and the number of rest of images
    face_images = 0
    rest_images = 0
    for image in data:
        if image[1] == 1:
            face_images += 1
        else:
            rest_images += 1

    # return the leaf node
    if face_images > rest_images:
        return 1
    else:
        return 0
    

# create a decision tree
def create_decision_tree_gini(data, max_depth, current_depth, min_samples_split):
    # find the best split
    best_feature, best_value = best_split(data)

    # if the best feature is None then return a leaf node
    if best_feature == None:
        return create_leaf(data)

    # if the current depth is equal to the maximum depth then return a leaf node
    if current_depth == max_depth:
        return create_leaf(data)
    
    # if the number of samples in the data is less than the minimum number of samples to split then return a leaf node
    if len(data) < min_samples_split:
        return create_leaf(data)

    # split the data
    left_data = []
    right_data = []
    for image in data:
        if image[0][best_feature] < best_value:
            left_data.append(image)
        else:
            right_data.append(image)

    # create the decision tree
    decision_tree = {}
    decision_tree["feature"] = best_feature
    decision_tree["value"] = best_value
    decision_tree["left"] = create_decision_tree_gini(left_data, max_depth, current_depth + 1, min_samples_split)
    decision_tree["right"] = create_decision_tree_gini(right_data, max_depth, current_depth + 1, min_samples_split)

    return decision_tree


#start time
start_time = time.time()

# create the decision tree
decision_tree_gini = create_decision_tree_gini(train_data, 10, 0, 7)

#end time
end_time = time.time()

# print the time taken to create the decision tree
print("Time taken to create the gini decision tree: " + str(end_time - start_time))






print("Gini Index Decision Tree:")


# validation 

# print the training accuracy, precision and recall of the decision tree
predictions = classify(train_data, decision_tree_gini)

print("Training Accuracy: " + str(accuracy(train_data, predictions)))
print("Training Precision: " + str(precision(train_data, predictions)))
print("Training Recall: " + str(recall(train_data, predictions)))

# print the training confusion matrix
print("Training Confusion Matrix:")
print(confusion_matrix(train_data, predictions))

# print the validation accuracy, precision and recall of the decision tree
predictions = classify(val_data, decision_tree_gini)

print("Validation Accuracy: " + str(accuracy(val_data, predictions)))
print("Validation Precision: " + str(precision(val_data, predictions)))
print("Validation Recall: " + str(recall(val_data, predictions)))

# print the validation confusion matrix
print("Validation Confusion Matrix:")
print(confusion_matrix(val_data, predictions))

# save the model to be used later
with open('decision_tree_gini.pkl', 'wb') as f:
    pickle.dump(decision_tree_gini, f)

# # load the model
# with open('decision_tree_gini.pkl', 'rb') as f:
#     decision_tree_gini = pickle.load(f)








