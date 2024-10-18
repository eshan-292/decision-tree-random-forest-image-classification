#take 3 paths as argument : 1 - train data path, 2 - test data path, 3 - output path

import sys

import getopt

import os

argv = sys.argv[1:]

opts, args = getopt.getopt(argv,"hi:o:",["train_path=","test_path=","out_path="])
for opt, arg in opts:
    if (opt == "--train_path"):
        train_data_path = arg
    elif (opt == "--test_path"):
        test_data_path = arg
    elif (opt == "--out_path"):
        output_path = arg


# train_data_path = sys.argv[1]
# test_data_path = sys.argv[2]
# output_path = sys.argv[3]







# Part 3.1 a

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

for filename in os.listdir(train_data_path + "/person"):
    # read the image
    # print("/content/drive/MyDrive/data/train/person/" + filename)
    
    
    image = cv2.imread(train_data_path+"/person/" + filename, 1)
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
for filename in os.listdir(train_data_path + "/airplane"):
    # read the image
    image = cv2.imread(train_data_path + "/airplane/" + filename, 1)

    if image is None:
      continue

    # resize the image
    # image = cv2.resize(image, (28, 28))

    # flatten the image
    image = image.flatten()

    # add the image to the list of images
    train_images_rest.append(image)

for filename in os.listdir(train_data_path + "/car"):
    # read the image
    image = cv2.imread(train_data_path + "/car/" + filename, 1)

    if image is None:
      continue

    # resize the image
    # image = cv2.resize(image, (28, 28))

    # flatten the image
    image = image.flatten()

    # add the image to the list of images
    train_images_rest.append(image)

for filename in os.listdir(train_data_path + "/cat"):
    # read the image
    image = cv2.imread(train_data_path + "/cat/" + filename, 1)

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





# # Validation 


# # read the face images from the data/validation directory
# val_images_face = []

# for filename in os.listdir("/content/drive/MyDrive/data/validation/person"):
#     # read the image
#     # print("/content/drive/MyDrive/data/train/person/" + filename)
    
    
#     image = cv2.imread("/content/drive/MyDrive/data/validation/person/" + filename, 1)
#     if image is None:
#       continue

#     # print(image.shape)

#     # resize the image
#     # image = cv2.resize(image, (28, 28))

#     # flatten the image
#     image = image.flatten()

#     # add the image to the list of images
#     val_images_face.append(image)

# # read the non-face images from the data/validation directory

# val_images_rest = []
# for filename in os.listdir("/content/drive/MyDrive/data/validation/airplane"):
#     # read the image
#     image = cv2.imread("/content/drive/MyDrive/data/validation/airplane/" + filename, 1)

#     if image is None:
#       continue

#     # resize the image
#     # image = cv2.resize(image, (28, 28))

#     # flatten the image
#     image = image.flatten()

#     # add the image to the list of images
#     val_images_rest.append(image)

# for filename in os.listdir("/content/drive/MyDrive/data/validation/car"):
#     # read the image
#     image = cv2.imread("/content/drive/MyDrive/data/validation/car/" + filename, 1)

#     if image is None:
#       continue

#     # resize the image
#     # image = cv2.resize(image, (28, 28))

#     # flatten the image
#     image = image.flatten()

#     # add the image to the list of images
#     val_images_rest.append(image)

# for filename in os.listdir("/content/drive/MyDrive/data/validation/dog"):
#     # read the image
#     image = cv2.imread("/content/drive/MyDrive/data/validation/dog/" + filename, 1)

#     if image is None:
#       continue

#     # resize the image
#     # image = cv2.resize(image, (28, 28))

#     # flatten the image
#     image = image.flatten()

#     # add the image to the list of images
#     val_images_rest.append(image)

# # create a list of labels for the face images
# val_labels_face = [1] * len(val_images_face)

# # create a list of labels for the rest of images
# val_labels_rest = [0] * len(val_images_rest)

# # create the validation data
# val_data = val_images_face + val_images_rest

# # create the validation labels
# val_labels = val_labels_face + val_labels_rest




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



#read the test data

# store the names of files in the test data
test_data_names = []

test_data = []
for filename in os.listdir(test_data_path):
    # read the image
    image = cv2.imread(test_data_path + "/" + filename, 1)

    if image is None:
      continue

    # resize the image
    # image = cv2.resize(image, (28, 28))

    # flatten the image
    image = image.flatten()

    # add the image to the list of images
    test_data.append(image)

    # add the name of the file without the extension to the list of names
    test_data_names.append(filename.split(".")[0])










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

    

  


# # calculate the accuracy, precision and recall



# def accuracy(data, predictions):
#     # count the number of correct predictions
#     correct_predictions = 0
#     for i in range(0, len(data)):
#         if data[i][1] == predictions[i]:
#             correct_predictions += 1

#     # return the accuracy
#     return correct_predictions / len(data)

# def precision(data, predictions):   
#     # count the number of true positives and the number of false positives
#     true_positives = 0
#     false_positives = 0
#     for i in range(0, len(data)):
#         if data[i][1] == 1 and predictions[i] == 1:
#             true_positives += 1
#         elif data[i][1] == 0 and predictions[i] == 1:
#             false_positives += 1

#     # return the precision
#     return true_positives / (true_positives + false_positives)

# def recall(data, predictions):
#     # count the number of true positives and the number of false negatives
#     true_positives = 0
#     false_negatives = 0
#     for i in range(0, len(data)):
#         if data[i][1] == 1 and predictions[i] == 1:
#             true_positives += 1
#         elif data[i][1] == 1 and predictions[i] == 0:
#             false_negatives += 1

#     # return the recall
#     return true_positives / (true_positives + false_negatives)

    

# # calculate the confusion matrix
# def confusion_matrix(data, predictions):
#     # count the number of true positives, false positives, true negatives and false negatives
#     true_positives = 0
#     false_positives = 0
#     true_negatives = 0
#     false_negatives = 0
#     for i in range(0, len(data)):
#         if data[i][1] == 1 and predictions[i] == 1:
#             true_positives += 1
#         elif data[i][1] == 0 and predictions[i] == 1:
#             false_positives += 1
#         elif data[i][1] == 0 and predictions[i] == 0:
#             true_negatives += 1
#         elif data[i][1] == 1 and predictions[i] == 0:
#             false_negatives += 1

#     # return the confusion matrix
#     return [[true_positives, false_positives], [false_negatives, true_negatives]]




# Train the model


# strat time
# start_time = time.time()

# build the decision tree
decision_tree = build_decision_tree(train_data, 10, 7)

# end time
# end_time = time.time()

# print the time
# print("Time to build the information gain decision tree: " + str(end_time - start_time))
# 





# print("Information Gain Decision Tree:")



# print the traininig accuracy, precision and recall of the decision tree on test data
predictions = classify(test_data, decision_tree)


#Your code should generate predicted output lables for test data (sample test data is provided in test sample folder) and save them into a csv file with image name (image id) as 1st column and predicted lables (labels) as 2nd column in the output path (do not include column headers in csv files)
#Generate a separate csv file for all required parts with name as test section.csv, for example : for 3.1.a part generate a csv file with name test 31a.csv

# write the predictions to a csv file
with open(output_path + "/test_31a.csv", "w") as csv_file:
    # write the header
    csv_file.write("image_id,label\n")

    # write the predictions
    for i in range(0, len(predictions)):
        #write the image name and the prediction
        csv_file.write(test_data_names[i] + "," + str(predictions[i]) + "\n")
        
        


















#3.1b - f

# build a binary decision tree classifier using the training data 

import numpy as np
import pandas as pd
import math
import random
import sys
import os
import cv2
import sklearn
import time
import matplotlib.pyplot as plt



# read the face images from the data/train directory
train_images_face = []

for filename in os.listdir(train_data_path + "/person"):
    # read the image
    # print("/content/drive/MyDrive/data/train/person/" + filename)
    
    
    image = cv2.imread(train_data_path + "/person/" + filename, 1)
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
for filename in os.listdir(train_data_path + "/airplane"):
    # read the image
    image = cv2.imread(train_data_path + "/airplane/" + filename, 1)

    if image is None:
      continue

    # resize the image
    # image = cv2.resize(image, (28, 28))

    # flatten the image
    image = image.flatten()

    # add the image to the list of images
    train_images_rest.append(image)

for filename in os.listdir(train_data_path + "/car"):
    # read the image
    image = cv2.imread(train_data_path + "/car/" + filename, 1)

    if image is None:
      continue

    # resize the image
    # image = cv2.resize(image, (28, 28))

    # flatten the image
    image = image.flatten()

    # add the image to the list of images
    train_images_rest.append(image)

for filename in os.listdir(train_data_path + "/cat"):
    # read the image
    image = cv2.imread(train_data_path + "/cat/" + filename, 1)

    if image is None:
      continue

    # resize the image
    # image = cv2.resize(image, (28, 28))

    # flatten the image
    image = image.flatten()

    # add the image to the list of images
    train_images_rest.append(image)


# create a list of labels for the face images
train_labels_face = [1] * len(train_images_face)

# create a list of labels for the rest of images
train_labels_rest = [0] * len(train_images_rest)

# create the training data
train_data = train_images_face + train_images_rest

# create the training labels
train_labels = train_labels_face + train_labels_rest



#read the test data
test_data = []

for filename in os.listdir(test_data_path):
    # read the image
    image = cv2.imread(test_data_path + "/" + filename, 1)

    if image is None:
      continue

    # resize the image
    # image = cv2.resize(image, (28, 28))

    # flatten the image
    image = image.flatten()

    # add the image to the list of images
    test_data.append(image)




# start the timer
# start_time = time.time()

# build the decision tree usin sklearn
from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=10, min_samples_split=7)
clf = clf.fit(train_data, train_labels)

# end the timer
# end_time = time.time()

# print the time taken to build the decision tree
# print("Training Time: ", end_time - start_time)



# # calculate accuracy, precision and recall on train data
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# train_pred = clf.predict(train_data)
# print("Training Accuracy: ", accuracy_score(train_labels, train_pred))
# print("Training Precision: ", precision_score(train_labels, train_pred))
# print("Training Recall: ", recall_score(train_labels, train_pred))

# # print the confusion matrix
# from sklearn.metrics import confusion_matrix
# print("Training Confusion Matrix: ")
# cm = confusion_matrix(train_labels, train_pred)
# print(cm)

# import matplotlib.pyplot as plt

# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()






# # test the decision tree using validation data
# val_pred = clf.predict(val_data)

# # calculate accuracy, precision and recall on validation data
# print("Validation Accuracy: ", accuracy_score(val_labels, val_pred))
# print("Validation Precision: ", precision_score(val_labels, val_pred))
# print("Validation Recall: ", recall_score(val_labels, val_pred))

# # print the confusion matrix
# from sklearn.metrics import confusion_matrix
# print("Validation Confusion Matrix: ")
# cm =confusion_matrix(val_labels, val_pred)
# print(cm)

# #visualise the confusion matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()




#predict the test data
test_pred = clf.predict(test_data)

#Your code should generate predicted output lables for test data (sample test data is provided in test sample folder) and save them into a csv file with image name (image id) as 1st column and predicted lables (labels) as 2nd column in the output path (do not include column headers in csv files)
#Generate a separate csv file for all required parts with name as test section.csv, for example : for 3.1.a part generate a csv file with name test 31a.csv

# save the predicted labels into a csv file
import csv
with open(output_path + "/test_31b.csv", "w") as f:
    
    writer = csv.writer(f)
    for i in range(len(test_pred)):
        
        # write the image name and the predicted label
        writer.writerow([test_data_names[i], test_pred[i]])

        










# save the original train data and validation data
train_data_orig = train_data
# val_data_orig = val_data
test_data_orig = test_data



#---------------------Decision Tree Grid-Search and Visualisation---------------------#


# select top-10 features from the data using sklearn feature_selection class and build a Decision Tree over those features
from sklearn.feature_selection import SelectKBest, chi2
selector = SelectKBest(chi2, k=10)
selector.fit(train_data, train_labels)
train_data = selector.transform(train_data)
# val_data = selector.transform(val_data)

# start the timer
# start_time = time.time()

# build the decision tree using sklearn
from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=10, min_samples_split=7)
clf = clf.fit(train_data, train_labels)

# end the timer
# end_time = time.time()

# print the time taken to build the decision tree

# print("Training Time in case of top 10 features: ", end_time - start_time)

# # visualize the decision tree
# import graphviz

# dot_data = tree.export_graphviz(clf, out_file=None, max_depth=10, feature_names=None, class_names=None, label='all', filled=True, leaves_parallel=False, impurity=True, node_ids=False, proportion=False, rotate=False, rounded=False, special_characters=False, precision=3)
# graph = graphviz.Source(dot_data)
# graph.render("decision_tree")

# # print the training accuracy
# print("Training Accuracy for top 10 features: ", clf.score(train_data, train_labels))

# #print the training precision
# from sklearn.metrics import precision_score
# print("Training Precision for top 10 features: ", precision_score(train_labels, clf.predict(train_data)))

# #print the training recall
# from sklearn.metrics import recall_score
# print("Training Recall for top 10 features: ", recall_score(train_labels, clf.predict(train_data)))

# #print the training confusion matrix
# from sklearn.metrics import confusion_matrix
# print("Training Confusion Matrix for top 10 features: ")
# cm = confusion_matrix(train_labels, clf.predict(train_data))
# print(cm)



# #visualize the confusion matrix

# import seaborn as sn
# import matplotlib.pyplot as plt

# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

# # print the validation accuracy
# print("Validation Accuracy for top 10 features: ", clf.score(val_data, val_labels))

# #print the validation precision
# from sklearn.metrics import precision_score
# print("Validation Precision for top 10 features: ", precision_score(val_labels, clf.predict(val_data)))

# #print the validation recall
# from sklearn.metrics import recall_score
# print("Validation Recall for top 10 features: ", recall_score(val_labels, clf.predict(val_data)))

# #print the validation confusion matrix
# from sklearn.metrics import confusion_matrix
# print("Validation Confusion Matrix for top 10 features: ")

# cm = confusion_matrix(val_labels, clf.predict(val_data))
# print(cm)

# #visualize the confusion matrix

# import seaborn as sn
# import matplotlib.pyplot as plt

# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()





# predict on the test data usign the top 10 features decision tree
test_data = selector.transform(test_data)
test_pred = clf.predict(test_data)

# save the predicted labels into a csv file
import csv
with open(output_path + "/test_31c.csv", "w") as f:
        
        writer = csv.writer(f)
        for i in range(len(test_pred)):
            # write the image name and the predicted label
            writer.writerow([test_data_names[i], test_pred[i]])














# print("Grid Search for Decision Tree:")





# # perform a grid search on the selected features to find the best hyperparameters
# from sklearn.model_selection import GridSearchCV
# parameters = {'criterion': ['gini', 'entropy'] , 'max_depth': [None, 5, 7, 10, 15], 'min_samples_split': [2, 4, 7, 9]}
# clf = GridSearchCV(tree.DecisionTreeClassifier(), parameters, cv=5)
# clf.fit(train_data, train_labels)

# # print the best hyperparameters
# print("Best Hyperparameters: ", clf.best_params_)

# # print the training accuracy
# print("Training Accuracy after performing grid search: ", clf.score(train_data, train_labels))

# #print the training precision
# from sklearn.metrics import precision_score
# print("Training Precision after performing grid search: ", precision_score(train_labels, clf.predict(train_data)))

# #print the training recall
# from sklearn.metrics import recall_score
# print("Training Recall after performing grid search: ", recall_score(train_labels, clf.predict(train_data)))

# #print the training confusion matrix
# from sklearn.metrics import confusion_matrix
# print("Training Confusion Matrix after performing grid search: ")
# cm = confusion_matrix(train_labels, clf.predict(train_data))
# print(cm)

# # visualize the confusion matrix
# import seaborn as sn
# import matplotlib.pyplot as plt

# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()


# # print the validation accuracy
# print("Validation Accuracy after performing grid search: ", clf.score(val_data, val_labels))

# #print the validation precision
# from sklearn.metrics import precision_score
# print("Validation Precision after performing grid search: ", precision_score(val_labels, clf.predict(val_data)))

# #print the validation recall
# from sklearn.metrics import recall_score
# print("Validation Recall after performing grid search: ", recall_score(val_labels, clf.predict(val_data)))

# #print the validation confusion matrix
# from sklearn.metrics import confusion_matrix
# print("Validation Confusion Matrix after performing grid search: ")
# cm = confusion_matrix(val_labels, clf.predict(val_data))
# print(cm)

# # visualize the confusion matrix
# import seaborn as sn
# import matplotlib.pyplot as plt

# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()






























#--------------------- Cost Complexity Pruning  ---------------------#



# restore the original train data and validation data
train_data = train_data_orig
# val_data = val_data_orig
test_data = test_data_orig


# use DecisionTreeClassifier.cost complexity pruning path to find the optimal value of alphas and the corresponding total leaf impurities at each step of the pruning process
from sklearn import tree
# clf = tree.DecisionTreeClassifier(random_state=0)



# path = clf.cost_complexity_pruning_path(train_data, train_labels)

# ccp_alphas, impurities = path.ccp_alphas, path.impurities



# # build a decision tree for each value of alpha and calculate the training and validation accuracy
# train_scores = []
# val_scores = []
# nodes = []
# depths = []


# for ccp_alpha in ccp_alphas:
#     clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
#     clf.fit(train_data, train_labels)
#     train_scores.append(clf.score(train_data, train_labels))
#     val_scores.append(clf.score(val_data, val_labels))
#     nodes.append(clf.tree_.node_count)
#     depths.append(clf.tree_.max_depth)

# # plot the total impurity of leaves vs the effective alphas of pruned tree

# fig, ax = plt.subplots()
# ax.set_xlabel("effective alpha")
# ax.set_ylabel("total impurity of leaves")
# ax.set_title("Total Impurity vs effective alpha for training set")
# ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', label="total_impurity", drawstyle="steps-post")
# ax.legend()
# plt.show()

# # Plot the number of nodes vs alpha and the depth of the tree vs alpha. The number of nodes and the depth of the tree will decrease as alpha increases.

# fig, ax = plt.subplots(2, 1)
# ax[0].set_xlabel("alpha")
# ax[0].set_ylabel("number of nodes")
# ax[0].set_title("Number of nodes vs alpha")
# ax[0].plot(ccp_alphas, nodes, marker='o', drawstyle="steps-post")

# ax[1].set_xlabel("alpha")
# ax[1].set_ylabel("depth of tree")
# ax[1].set_title("Depth vs alpha")
# ax[1].plot(ccp_alphas, depths, marker='o', drawstyle="steps-post")
# plt.show()

# # plot the training and validation accuracy vs alpha
# fig, ax = plt.subplots()
# ax.set_xlabel("alpha")
# ax.set_ylabel("accuracy")
# ax.set_title("Accuracy vs alpha for training and validation sets")
# ax.plot(ccp_alphas, train_scores, marker='o', label="train", drawstyle="steps-post")
# ax.plot(ccp_alphas, val_scores, marker='o', label="val", drawstyle="steps-post")
# ax.legend()
# plt.show()

# # Use the validation split to determine the best-performing tree and report the training and validation accuracy for the best tree
# # best_alpha = ccp_alphas[np.argmax(val_scores)]
best_alpha = 0.0027352941176470593
clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=best_alpha)
clf.fit(train_data, train_labels)

# print("Best alpha: ", best_alpha)
# print("Best performing tree statistics:")

# print("Best training accuracy: ", clf.score(train_data, train_labels))
# print("Best training precision: ", precision_score(train_labels, clf.predict(train_data)))
# print("Best training recall: ", recall_score(train_labels, clf.predict(train_data)))

# # print the confusion matrix
# from sklearn.metrics import confusion_matrix
# print("Training Confusion Matrix: ")
# cm = confusion_matrix(train_labels, clf.predict(train_data))
# print(cm)

# # visualise the confusion matrix
# import seaborn as sns
# from sklearn.metrics import confusion_matrix
# # cm = confusion_matrix(train_labels, clf.predict(train_data))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()


# print("Best validation accuracy: ", clf.score(val_data, val_labels))
# print("Best validation precision: ", precision_score(val_labels, clf.predict(val_data)))
# print("Best validation recall: ", recall_score(val_labels, clf.predict(val_data)))

# print("Validation Confusion Matrix: ")
# cm = confusion_matrix(val_labels, clf.predict(val_data))
# print(cm)


# # visualise the confusion matrix
# import seaborn as sns
# from sklearn.metrics import confusion_matrix
# # cm = confusion_matrix(val_labels, clf.predict(val_data))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()



# # Visualize the best-pruned tree
# dot_data = tree.export_graphviz(clf, out_file=None, max_depth=10, feature_names=None, class_names=None, label='all', filled=True, leaves_parallel=False, impurity=True, node_ids=False, proportion=False, rotate=False, rounded=False, special_characters=False, precision=3)
# graph = graphviz.Source(dot_data)
# graph.render("decision_tree_pruned")




# save the predicitions on test data
test_predictions = clf.predict(test_data)

# save the predicted labels into a csv file
import csv
with open(output_path + "/test_31d.csv", "w") as f:
        
        writer = csv.writer(f)
        for i in range(len(test_pred)):
            # write the image name and the predicted label
            writer.writerow([test_data_names[i], test_pred[i]])














#--------------------- Random Forests ---------------------#



# print("Random Forests: ")





# use default hyperparameters to build a random forest using sklearn
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=7, random_state=0)
clf.fit(train_data, train_labels)

# # print the training and validation accuracy, precision and recall
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# print("Training Accuracy: ", accuracy_score(train_labels, clf.predict(train_data)))
# print("Training Precision: ", precision_score(train_labels, clf.predict(train_data)))
# print("Training Recall: ", recall_score(train_labels, clf.predict(train_data)))

# print("Validation Accuracy: ", accuracy_score(val_labels, clf.predict(val_data)))
# print("Validation Precision: ", precision_score(val_labels, clf.predict(val_data)))
# print("Validation Recall: ", recall_score(val_labels, clf.predict(val_data)))

# # perform a grid search on the selected features to find the best hyperparameters
# from sklearn.model_selection import GridSearchCV

# parameters = {'n_estimators': [80, 100, 150, 200], 'criterion': ['gini', 'entropy'] , 'max_depth': [None, 5, 7, 10], 'min_samples_split': [5, 7, 10]}
# clf = GridSearchCV(RandomForestClassifier(), parameters, cv=5)
# clf.fit(train_data, train_labels)

# # print the best hyperparameters
# print("Best Hyperparameters: ", clf.best_params_)

# # print the training accuracy, precision and recall
# print("Training Accuracy: ", clf.best_score_)
# print("Training Precision: ", precision_score(train_labels, clf.predict(train_data)))
# print("Training Recall: ", recall_score(train_labels, clf.predict(train_data)))

# #print the training confusion matrix
# from sklearn.metrics import confusion_matrix
# print("Training Confusion Matrix: ")
# print(confusion_matrix(train_labels, clf.predict(train_data)))

# # visialized the training confusion matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(train_labels, clf.predict(train_data))
# ax = plt.subplot()
# sns.heatmap(cm, annot=True, ax = ax, fmt='g', cmap='Blues')



# # print the validation accuracy, precision and recall
# print("Validation Accuracy: ", clf.score(val_data, val_labels))
# print("Validation Precision: ", precision_score(val_labels, clf.predict(val_data)))
# print("Validation Recall: ", recall_score(val_labels, clf.predict(val_data)))

# #print the validation confusion matrix
# print("Validation Confusion Matrix: ")
# print(confusion_matrix(val_labels, clf.predict(val_data)))



# # visualize the confusion matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(val_labels, clf.predict(val_data))
# ax = plt.subplot()
# sns.heatmap(cm, annot=True, ax = ax, fmt='g', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()




# # use grid search to find the best hyperparameters for the random forest

# # define the parameter values that should be searched
# parameters = {'n_estimators': [80, 100, 150, 200], 'criterion': ['gini', 'entropy'] , 'max_depth': [None, 5, 7, 10], 'min_samples_split': [5, 7, 10]}

# # instantiate the grid
# from sklearn.model_selection import GridSearchCV
# clf = GridSearchCV(RandomForestClassifier(), parameters, cv=5)

# # fit the grid with data
# clf.fit(train_data, train_labels)


# # print the best hyperparameters
# print("Best Hyperparameters: ", clf.best_params_)

# # use the best hyperparameters to build a random forest using sklearn
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=clf.best_params_['n_estimators'], max_depth=clf.best_params_['max_depth'], min_samples_split=clf.best_params_['min_samples_split'], criterion=clf.best_params_['criterion'], random_state=0)
# clf.fit(train_data, train_labels)


# # print the training and validation accuracy, precision and recall
# from sklearn.metrics import accuracy_score, precision_score, recall_score

# print("Training Accuracy: ", accuracy_score(train_labels, clf.predict(train_data)))
# print("Training Precision: ", precision_score(train_labels, clf.predict(train_data)))
# print("Training Recall: ", recall_score(train_labels, clf.predict(train_data)))

# print("Validation Accuracy: ", accuracy_score(val_labels, clf.predict(val_data)))
# print("Validation Precision: ", precision_score(val_labels, clf.predict(val_data)))
# print("Validation Recall: ", recall_score(val_labels, clf.predict(val_data)))

# # print the training and validation confusion matrices
# from sklearn.metrics import confusion_matrix
# print("Training Confusion Matrix: ")
# print(confusion_matrix(train_labels, clf.predict(train_data)))


# print("Validation Confusion Matrix: ")
# print(confusion_matrix(val_labels, clf.predict(val_data)))





# # visualise the confusion matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# cm = confusion_matrix(val_labels, clf.predict(val_data))
# ax= plt.subplot()
# sns.heatmap(cm, annot=True, ax = ax, fmt='g'); #annot=True to annotate cells
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()






# predict the labels of the test data
test_pred = clf.predict(test_data)


# write the predictions to a csv file

# save the predicted labels into a csv file
import csv
with open(output_path + "/test_31e.csv", "w") as f:
        
        writer = csv.writer(f)
        for i in range(len(test_pred)):
            # write the image name and the predicted label
            writer.writerow([test_data_names[i], test_pred[i]])






















#--------------------- Gradient Boosting ---------------------#








# print("Gradient Boosting: ")


# # Implement a Gradient Boosted Classifier using sklearn.ensemble.GradientBoostingClassifier by performing a grid seacrh 

# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import precision_score, recall_score


# # train the gradient boosting classifier
# clf = GradientBoostingClassifier()
# clf.fit(train_data, train_labels)

# # print the training accuracy, precision and recall
# print("Gradient Boosting Training Accuracy: ", clf.score(train_data, train_labels))
# print("Gradient Boosting Training Precision: ", precision_score(train_labels, clf.predict(train_data)))
# print("Gradient Boosting Training Recall: ", recall_score(train_labels, clf.predict(train_data)))


# # print the validation accuracy, precision and recall
# print("Gradient Boosting Validation Accuracy: ", clf.score(val_data, val_labels))
# print("Gradient Boosting Validation Precision: ", precision_score(val_labels, clf.predict(val_data)))
# print("Gradient Boosting Validation Recall: ", recall_score(val_labels, clf.predict(val_data)))

# # print the training and validation confusion matrices
# from sklearn.metrics import confusion_matrix
# print("Gradient Boosting Training Confusion Matrix: ")
# cm = confusion_matrix(train_labels, clf.predict(train_data))
# print(cm)

# # visualize the training confusion matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# ax = plt.subplot()
# sns.heatmap(cm, annot=True, ax = ax, fmt='g', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()


# print("Gradient Boosting Validation Confusion Matrix: ")
# cm = confusion_matrix(val_labels, clf.predict(val_data))
# print(cm)

# # visualize the validation confusion matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# ax = plt.subplot()
# sns.heatmap(cm, annot=True, ax = ax, fmt='g', cmap='Blues')

# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()



# # perform a grid search on the selected features to find the best hyperparameters
# parameters = {'n_estimators': [20, 30, 40, 50] , 'max_depth': [5, 6, 7, 8, 9, 10], 'subsample': [0.2, 0.3, 0.4, 0.5, 0.6]}
# clf = GridSearchCV(GradientBoostingClassifier(), parameters, cv=5)
# start = time.time()
# clf.fit(train_data, train_labels)
# end = time.time()



# # print the time
# print("Gradient Boosting Training Time: ", end - start)


# # print the best hyperparameters
# print("Gradient Boosting Best Hyperparameters: ", clf.best_params_)

# # print the training accuracy, precision and recall
# print("Gradient Boosting Training Accuracy: ", clf.best_score_)
# print("Gradient Boosting Training Precision: ", precision_score(train_labels, clf.predict(train_data)))
# print("Gradient Boosting Training Recall: ", recall_score(train_labels, clf.predict(train_data)))

# #print the training confusion matrix
# from sklearn.metrics import confusion_matrix
# print("Gradient Boosting Training Confusion Matrix: ")
# cm = confusion_matrix(train_labels, clf.predict(train_data))
# print(cm)



# # visualise the training confusion matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# ax = plt.subplot()
# sns.heatmap(cm, annot=True, ax = ax, fmt='g', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()


# # print the validation accuracy, precision and recall
# print("Gradient Boosting Validation Accuracy: ", clf.score(val_data, val_labels))
# print("Gradient Boosting Validation Precision: ", precision_score(val_labels, clf.predict(val_data)))
# print("Gradient Boosting Validation Recall: ", recall_score(val_labels, clf.predict(val_data)))

# #print the validation confusion matrix
# print("Gradient Boosting Validation Confusion Matrix: ")
# cm = confusion_matrix(val_labels, clf.predict(val_data))
# print(cm)

# # visualise the validation confusion matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# ax = plt.subplot()
# sns.heatmap(cm, annot=True, ax = ax, fmt='g', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()










#--------------------- Extreme Gradient Boosting XGBoost ---------------------#

# print("Extreme Gradient Boosting XGBoost: ")

# XGBoost Classifier
import xgboost as xgb
from xgboost import XGBClassifier

# train the xgboost classifier
xgb_clf = XGBClassifier()
xgb_clf.fit(train_data, train_labels)

# # print the training and validation accuracy, precision and recall
# from sklearn.metrics import accuracy_score, precision_score, recall_score

# print("XGBoost Training Accuracy: ", accuracy_score(train_labels, xgb_clf.predict(train_data)))
# print("XGBoost Training Precision: ", precision_score(train_labels, xgb_clf.predict(train_data)))
# print("XGBoost Training Recall: ", recall_score(train_labels, xgb_clf.predict(train_data)))

# # print the training and validation confusion matrices
# from sklearn.metrics import confusion_matrix
# print("XGBoost Training Confusion Matrix: ")
# cm = confusion_matrix(train_labels, xgb_clf.predict(train_data))
# print(cm)

# # visualize the validation confusion matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# ax = plt.subplot()
# sns.heatmap(cm, annot=True, ax = ax, fmt='g', cmap='Blues')

# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()




# print("XGBoost Validation Accuracy: ", accuracy_score(val_labels, xgb_clf.predict(val_data)))
# print("XGBoost Validation Precision: ", precision_score(val_labels, xgb_clf.predict(val_data)))
# print("XGBoost Validation Recall: ", recall_score(val_labels, xgb_clf.predict(val_data)))

# # print the validation confusion matrix
# print("XGBoost Validation Confusion Matrix: ")


# cm = confusion_matrix(val_labels, xgb_clf.predict(val_data))
# print(cm)

# # visualize the validation confusion matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# ax = plt.subplot()
# sns.heatmap(cm, annot=True, ax = ax, fmt='g', cmap='Blues')

# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()




# # perform a grid search on the selected features to find the best hyperparameters
# from sklearn.model_selection import GridSearchCV

# parameters = {'n_estimators': [20, 30, 40, 50] , 'max_depth': [5, 6, 7, 8, 9, 10], 'subsample': [0.2, 0.3, 0.4, 0.5, 0.6]}
# clf = GridSearchCV(XGBClassifier(), parameters, cv=5)

# # start the time
# start = time.time()

# clf.fit(train_data, train_labels)

# # end the time
# end = time.time()

# # print the time
# print("XGBoost Training Time: ", end - start)

# # print the best hyperparameters
# print("XGBoost Best Hyperparameters: ", clf.best_params_)
# # print the training accuracy, precision and recall
# print("XGBoost Training Accuracy: ", clf.best_score_)
# print("XGBoost Training Precision: ", precision_score(train_labels, clf.predict(train_data)))
# print("XGBoost Training Recall: ", recall_score(train_labels, clf.predict(train_data)))

# #print the training confusion matrix
# from sklearn.metrics import confusion_matrix
# print("XGBoost Training Confusion Matrix: ")
# cm = confusion_matrix(train_labels, clf.predict(train_data))
# print(cm)

# # visualise the training confusion matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# ax = plt.subplot()
# sns.heatmap(cm, annot=True, ax = ax, fmt='g', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()


# # print the validation accuracy, precision and recall
# print("XGBoost Validation Accuracy: ", clf.score(val_data, val_labels))
# print("XGBoost Validation Precision: ", precision_score(val_labels, clf.predict(val_data)))
# print("XGBoost Validation Recall: ", recall_score(val_labels, clf.predict(val_data)))

# #print the validation confusion matrix
# print("XGBoost Validation Confusion Matrix: ")
# cm = confusion_matrix(val_labels, clf.predict(val_data))
# print(cm)

# # visualise the validation confusion matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# ax = plt.subplot()
# sns.heatmap(cm, annot=True, ax = ax, fmt='g', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()



# predict obn the test data

test_pred = xgb_clf.predict(test_data)


# save the predicted labels into a csv file
import csv
with open(output_path + "/test_31f.csv", "w") as f:
        
        writer = csv.writer(f)
        for i in range(len(test_pred)):
            # write the image name and the predicted label
            writer.writerow(test_data_names[i], test_pred[i])









# Competetive part



# save the predicted labels into a csv file
import csv
with open(output_path + "/test_31h.csv", "w") as f:
        
        writer = csv.writer(f)
        for i in range(len(test_pred)):
            # write the image name and the predicted label
            writer.writerow(test_data_names[i], test_pred[i])




