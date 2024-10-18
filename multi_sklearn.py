# build a multiclass decision tree classifier using sklearn

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


# Label the classes as given : Cars - 0, Faces - 1, Airplanes - 2, Dogs - 3.

# create the training labels
train_labels = [1] * len(train_images_person) + [0] * len(train_images_car) + [2] * len(train_images_airplane) + [3] * len(train_images_dog)

# create the training data
train_data = train_images_person + train_images_car + train_images_airplane + train_images_dog

# Build the decision tree classifier using the training data and labels
# Use the default parameters for the decision tree classifier
# Use random_state=0 for reproducibility
# Use max_depth=10
# Use min_samples_split=7






# Validation 


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

# read the car images from the data/validation directory
val_images_car = []

for filename in os.listdir("/content/drive/MyDrive/data/validation/car"):
    # read the image
    # print("/content/drive/MyDrive/data/train/car/" + filename)
    
    
    image = cv2.imread("/content/drive/MyDrive/data/validation/car/" + filename, 1)
    if image is None:
      continue

    # print(image.shape)

    # resize the image
    # image = cv2.resize(image, (28, 28))

    # flatten the image
    image = image.flatten()

    # add the image to the list of images
    val_images_car.append(image)

# read the airplane images from the data/validation directory
val_images_airplane = []

for filename in os.listdir("/content/drive/MyDrive/data/validation/airplane"):
    # read the image
    # print("/content/drive/MyDrive/data/train/airplane/" + filename)
    
    
    image = cv2.imread("/content/drive/MyDrive/data/validation/airplane/" + filename, 1)
    if image is None:
      continue

    # print(image.shape)

    # resize the image
    # image = cv2.resize(image, (28, 28))

    # flatten the image
    image = image.flatten()

    # add the image to the list of images
    val_images_airplane.append(image)

# read the dog images from the data/validation directory
val_images_dog = []

for filename in os.listdir("/content/drive/MyDrive/data/validation/dog"):
    # read the image
    # print("/content/drive/MyDrive/data/train/dog/" + filename)
    
    
    image = cv2.imread("/content/drive/MyDrive/data/validation/dog/" + filename, 1)
    if image is None:
      continue

    # print(image.shape)

    # resize the image
    # image = cv2.resize(image, (28, 28))

    # flatten the image
    image = image.flatten()

    # add the image to the list of images
    val_images_dog.append(image)





# Label the classes as given : Cars - 0, Faces - 1, Airplanes - 2, Dogs - 3.

# create the validation labels
val_labels = [1] * len(val_images_face) + [0] * len(val_images_car) + [2] * len(val_images_airplane) + [3] * len(val_images_dog)

# create the validation data
val_data = val_images_face + val_images_car + val_images_airplane + val_images_dog









#Train the model


# start the timer
start_time = time.time()


from sklearn.tree import DecisionTreeClassifier


# create the decision tree classifier
clf = DecisionTreeClassifier(random_state=0, max_depth=10, min_samples_split=7)



# train the classifier
clf.fit(train_data, train_labels)

# stop the timer
end_time = time.time()

# print the training time
print("Training time: ", end_time - start_time)






# Use the trained classifier to predict the labels for the validation data
# Store the predicted labels in a list named val_predictions

val_predictions = clf.predict(val_data)

# Print the accuracy, precision and recall for training data

from sklearn.metrics import accuracy_score, precision_score, recall_score

# calculate the accuracy
accuracy = accuracy_score(val_labels, val_predictions)

# calculate the precision
precision = precision_score(val_labels, val_predictions, average='macro')

# calculate the recall
recall = recall_score(val_labels, val_predictions, average='macro')

# print the accuracy, precision and recall
print("Training Accuracy: ", accuracy)
print("Training Precision: ", precision)
print("Training Recall: ", recall)\

# Print the confusion matrix for training data

from sklearn.metrics import confusion_matrix

# calculate the confusion matrix
cm = confusion_matrix(val_labels, val_predictions)

# print the confusion matrix
print("Training Confusion Matrix: ")
print(cm)

#visualize the confusion matrix

import seaborn as sns
import matplotlib.pyplot as plt

# plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# set the labels
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# show the plot
plt.show()



# Print the accuracy, precision and recall for validation data

# calculate the accuracy
accuracy = accuracy_score(val_labels, val_predictions)

# calculate the precision
precision = precision_score(val_labels, val_predictions, average='macro')

# calculate the recall
recall = recall_score(val_labels, val_predictions, average='macro')

# print the accuracy, precision and recall
print("Validation Accuracy: ", accuracy)
print("Validation Precision: ", precision)
print("Validation Recall: ", recall)

# Print the confusion matrix for validation data

from sklearn.metrics import confusion_matrix

# calculate the confusion matrix
cm = confusion_matrix(val_labels, val_predictions)

# print the confusion matrix
print("Validation Confusion Matrix: ")
print(cm)

# plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# set the labels
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# show the plot
plt.show()




#save the original train data and validation data
train_data_original = train_data
val_data_original = val_data







#---------------------Decision Tree Grid-Search and Visualisation---------------------#

print("Decision Tree Grid-Search and Visualisation:")



# select top-10 features from the data using sklearn feature_selection class and build a Decision Tree over those features

# start the timer
start_time = time.time()


from sklearn.feature_selection import SelectKBest, chi2

# select the top-10 features
selector = SelectKBest(chi2, k=10)

# fit the selector to the training data
selector.fit(train_data, train_labels)

# transform the training data
train_data = selector.transform(train_data)

# transform the validation data
val_data = selector.transform(val_data)

# stop the timer
end_time = time.time()

# print the training time
print("Feature Selection Time: ", end_time - start_time)

# start the timer
start_time = time.time()

# create the decision tree classifier
clf = DecisionTreeClassifier()

# fit the classifier to the training data
clf.fit(train_data, train_labels)

# stop the timer
end_time = time.time()

# print the training time
print("Training Time for top 10 features: ", end_time - start_time)

# Visualize the decision tree

from sklearn import tree


# visualize the decision tree
tree.plot_tree(clf)





# predict the labels for the validation data
val_predictions = clf.predict(val_data)


# Print the accuracy, precision and recall for training data

from sklearn.metrics import accuracy_score, precision_score, recall_score

# calculate the accuracy
accuracy = accuracy_score(val_labels, val_predictions)

# calculate the precision
precision = precision_score(val_labels, val_predictions, average='macro')

# calculate the recall
recall = recall_score(val_labels, val_predictions, average='macro')

# print the accuracy, precision and recall
print("Training Accuracy for top 10 features: ", accuracy)
print("Training Precision for top 10 features: ", precision)
print("Training Recall for top 10 features: ", recall)

# Print the confusion matrix for training data

from sklearn.metrics import confusion_matrix

# calculate the confusion matrix
cm = confusion_matrix(val_labels, val_predictions)

# print the confusion matrix
print("Training Confusion Matrix for top 10 features: ")
print(cm)

# plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# set the labels
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# show the plot
plt.show()


# Print the accuracy, precision and recall for validation data

# calculate the accuracy
accuracy = accuracy_score(val_labels, val_predictions)

# calculate the precision
precision = precision_score(val_labels, val_predictions, average='macro')   

# calculate the recall
recall = recall_score(val_labels, val_predictions, average='macro')

# print the accuracy, precision and recall
print("Validation Accuracy for top 10 features: ", accuracy)
print("Validation Precision for top 10 features: ", precision)
print("Validation Recall for top 10 features: ", recall)

# Print the confusion matrix for validation data

from sklearn.metrics import confusion_matrix

# calculate the confusion matrix
cm = confusion_matrix(val_labels, val_predictions)

# print the confusion matrix
print("Validation Confusion Matrix for top 10 features: ")
print(cm)


# plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# set the labels
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# show the plot
plt.show()









# perform a grid search on the selected features to find the best hyperparameters

from sklearn.model_selection import GridSearchCV
parameters = {'criterion': ['gini', 'entropy'] , 'max_depth': [None, 5, 7, 10, 15], 'min_samples_split': [2, 4, 7, 9]}

# start the timer
start_time = time.time()

# create the decision tree classifier
clf = DecisionTreeClassifier()

# create the grid search
grid_search = GridSearchCV(clf, parameters, cv=5)

# fit the grid search to the training data
grid_search.fit(train_data, train_labels)

# stop the timer
end_time = time.time()

# print the training time
print("Training Time for top 10 features with Grid Search: ", end_time - start_time)

# print the best parameters
print("Best Parameters: ", grid_search.best_params_)



# predict the labels for the validation data
val_predictions = grid_search.predict(val_data)

# Print the accuracy, precision and recall for training data

from sklearn.metrics import accuracy_score, precision_score, recall_score

# calculate the accuracy
accuracy = accuracy_score(val_labels, val_predictions)

# calculate the precision
precision = precision_score(val_labels, val_predictions, average='macro')

# calculate the recall
recall = recall_score(val_labels, val_predictions, average='macro')

# print the accuracy, precision and recall

print("Training Accuracy for top 10 features with Grid Search: ", accuracy)
print("Training Precision for top 10 features with Grid Search: ", precision)
print("Training Recall for top 10 features with Grid Search: ", recall)

# Print the confusion matrix for training data

from sklearn.metrics import confusion_matrix

# calculate the confusion matrix
cm = confusion_matrix(val_labels, val_predictions)

# print the confusion matrix
print("Training Confusion Matrix for top 10 features with Grid Search: ")

print(cm)


# plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# set the labels
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# show the plot
plt.show()

# Print the accuracy, precision and recall for validation data

# calculate the accuracy
accuracy = accuracy_score(val_labels, val_predictions)


# calculate the precision

precision = precision_score(val_labels, val_predictions, average='macro')

# calculate the recall
recall = recall_score(val_labels, val_predictions, average='macro')

# print the accuracy, precision and recall
print("Validation Accuracy for top 10 features with Grid Search: ", accuracy)
print("Validation Precision for top 10 features with Grid Search: ", precision)
print("Validation Recall for top 10 features with Grid Search: ", recall)

# Print the confusion matrix for validation data

from sklearn.metrics import confusion_matrix

# calculate the confusion matrix
cm = confusion_matrix(val_labels, val_predictions)

# print the confusion matrix
print("Validation Confusion Matrix for top 10 features with Grid Search: ")
print(cm)



# plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# set the labels
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# show the plot
plt.show()















#--------------------- Cost Complexity Pruning  ---------------------#


print("Cost Complexity Pruning:")

#restore the original data
train_data = train_data_original
val_data = val_data_original


# use DecisionTreeClassifier.cost complexity pruning path to find the optimal value of alphas and the corresponding total leaf impurities at each step of the pruning process
from sklearn import tree
clf = tree.DecisionTreeClassifier(random_state=0)

path = clf.cost_complexity_pruning_path(train_data, train_labels)

# plot the total leaf impurities against the values of alpha
ccp_alphas, impurities = path.ccp_alphas, path.impurities
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")


# build a decision tree for each value of alpha and calculate the training and validation accuracy
train_scores = []
val_scores = []
nodes = []
depths = []

for ccp_alpha in ccp_alphas:
    clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(train_data, train_labels)
    train_scores.append(clf.score(train_data, train_labels))
    val_scores.append(clf.score(val_data, val_labels))
    nodes.append(clf.tree_.node_count)
    depths.append(clf.get_depth())

# Plot the number of nodes vs alpha and the depth of the tree vs alpha. The number of nodes and the depth of the tree will decrease as alpha increases.
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, nodes, marker='o', drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")


# plot the depth of the tree vs alpha
ax[1].plot(ccp_alphas, depths, marker='o', drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
plt.show()


# plot the training and validation accuracy vs alpha
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and validation sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, val_scores, marker='o', label="validation",
        drawstyle="steps-post")
ax.legend()
plt.show()







# Use the validation split to determine the best-performing tree and report the training and validation accuracy for the best tree

# find the index of the best alpha
index = np.argmax(val_scores)

# print the best alpha
print("Best alpha: ", ccp_alphas[index])

print("Best performing tree statistics:")

# train the decision tree classifier with the best alpha
clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alphas[index])
clf.fit(train_data, train_labels)

# print the training accuracy
print("Training Accuracy for best performing tree: ", clf.score(train_data, train_labels))


from sklearn.metrics import precision_score, recall_score

#print the precision and recall for training data
print("Training Precision for best performing tree: ", precision_score(train_labels, clf.predict(train_data), average='macro'))
print("Training Recall for best performing tree: ", recall_score(train_labels, clf.predict(train_data), average='macro'))

#print the confusion matrix for training data
from sklearn.metrics import confusion_matrix
print("Training Confusion Matrix for best performing tree: ")
cm = confusion_matrix(train_labels, clf.predict(train_data))
print(cm)



# plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# set the labels
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# show the plot
plt.show()


#predict the labels for the validation data
val_predictions = clf.predict(val_data)

# print the validation accuracy
print("Validation Accuracy for best performing tree: ", clf.score(val_data, val_labels))

#print the precision and recall for validation data
print("Validation Precision for best performing tree: ", precision_score(val_labels, val_predictions, average='macro'))

print("Validation Recall for best performing tree: ", recall_score(val_labels, val_predictions, average='macro'))

#print the confusion matrix for validation data
from sklearn.metrics import confusion_matrix
print("Validation Confusion Matrix for best performing tree: ")
cm = confusion_matrix(val_labels, val_predictions)
print(cm)


# visualise the confusion matrix

# plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# set the labels
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# show the plot
plt.show()



# visualize the tree
from sklearn import tree
plt.figure(figsize=(25,20))
tree.plot_tree(clf, filled=True)
plt.show()














#--------------------- Random Forest ---------------------#


print("Random Forest:")

# use default hyperparameters to build a random forest using sklearn
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
clf.fit(train_data, train_labels)

# print the training accuracy
print("Training Accuracy for random forest: ", clf.score(train_data, train_labels))

#print the precision and recall for training data
# print("Training Precision for random forest: ", precision_score(train_labels, clf.predict(train_data), average='macro'))
# print("Training Recall for random forest: ", recall_score(train_labels, clf.predict(train_data), average='macro'))

#print the confusion matrix for training data
from sklearn.metrics import confusion_matrix
print("Training Confusion Matrix for random forest: ")
cm = confusion_matrix(train_labels, clf.predict(train_data))
print(cm)

# plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# set the labels
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# show the plot
plt.show()



#predict the labels for the validation data
val_predictions = clf.predict(val_data)

# print the validation accuracy
print("Validation Accuracy for random forest: ", clf.score(val_data, val_labels))


#print the precision and recall for validation data
# print("Validation Precision for random forest: ", precision_score(val_labels, val_predictions, average='macro'))

# print("Validation Recall for random forest: ", recall_score(val_labels, val_predictions, average='macro'))

#print the confusion matrix for validation data
from sklearn.metrics import confusion_matrix
print("Validation Confusion Matrix for random forest: ")
cm = confusion_matrix(val_labels, val_predictions)

print(cm)


# plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# set the labels
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# show the plot
plt.show()





# use grid search to find the best hyperparameters for the random forest

# define the parameter values that should be searched
parameters = {'n_estimators': [80, 100, 150, 200], 'criterion': ['gini', 'entropy'] , 'max_depth': [None, 5, 7, 10], 'min_samples_split': [5, 7, 10]}

# instantiate the grid
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(clf, parameters, cv=5)

# fit the grid with data
grid.fit(train_data, train_labels)

# view the complete results
print("Grid Search Results:")
print(grid.cv_results_)
print("Best Parameters: ", grid.best_params_)
# print("Best Score: ", grid.best_score_)

# use the best hyperparameters to build a random forest using sklearn
clf = RandomForestClassifier(random_state=0, n_estimators=grid.best_params_['n_estimators'], criterion=grid.best_params_['criterion'], max_depth=grid.best_params_['max_depth'], min_samples_split=grid.best_params_['min_samples_split'])

clf.fit(train_data, train_labels)

# print the training accuracy
print("Training Accuracy for random forest with best hyperparameters: ", clf.score(train_data, train_labels))

#print the precision and recall for training data
print("Training Precision for random forest with best hyperparameters: ", precision_score(train_labels, clf.predict(train_data), average='macro'))

print("Training Recall for random forest with best hyperparameters: ", recall_score(train_labels, clf.predict(train_data), average='macro'))

#print the confusion matrix for training data
from sklearn.metrics import confusion_matrix

print("Training Confusion Matrix for random forest with best hyperparameters: ")

cm = confusion_matrix(train_labels, clf.predict(train_data))
print(cm)

# plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# set the labels
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# show the plot
plt.show()




#predict the labels for the validation data
val_predictions = clf.predict(val_data)

# print the validation accuracy
print("Validation Accuracy for random forest with best hyperparameters: ", clf.score(val_data, val_labels))

#print the precision and recall for validation data
print("Validation Precision for random forest with best hyperparameters: ", precision_score(val_labels, val_predictions, average='macro'))

print("Validation Recall for random forest with best hyperparameters: ", recall_score(val_labels, val_predictions, average='macro'))

#print the confusion matrix for validation data
from sklearn.metrics import confusion_matrix
print("Validation Confusion Matrix for random forest with best hyperparameters: ")


cm = confusion_matrix(val_labels, val_predictions)
print(cm)



# plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# set the labels
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# show the plot
plt.show()





























#--------------------- Gradient Boosting ---------------------#






print("Gradient Boosting: ")


# Implement a Gradient Boosted Classifier using sklearn.ensemble.GradientBoostingClassifier by performing a grid seacrh 

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score





# train the gradient boosting classifier
clf = GradientBoostingClassifier()
clf.fit(train_data, train_labels)

# print the training accuracy, precision and recall
print("Gradient Boosting Training Accuracy: ", clf.score(train_data, train_labels))
print("Gradient Boosting Training Precision: ", precision_score(train_labels, clf.predict(train_data)), average='macro')
print("Gradient Boosting Training Recall: ", recall_score(train_labels, clf.predict(train_data), average='macro'))


# print the validation accuracy, precision and recall
print("Gradient Boosting Validation Accuracy: ", clf.score(val_data, val_labels))
print("Gradient Boosting Validation Precision: ", precision_score(val_labels, clf.predict(val_data), average='macro'))
print("Gradient Boosting Validation Recall: ", recall_score(val_labels, clf.predict(val_data), average='macro'))

# print the training and validation confusion matrices
from sklearn.metrics import confusion_matrix
print("Gradient Boosting Training Confusion Matrix: ")
cm = confusion_matrix(train_labels, clf.predict(train_data), )
print(cm)

# visualize the training confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


print("Gradient Boosting Validation Confusion Matrix: ")
cm = confusion_matrix(val_labels, clf.predict(val_data))
print(cm)

# visualize the validation confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g', cmap='Blues')

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

















# perform a grid search on the selected features to find the best hyperparameters
parameters = {'n_estimators': [20, 30, 40, 50] , 'max_depth': [5, 6, 7, 8, 9, 10], 'subsample': [0.2, 0.3, 0.4, 0.5, 0.6]}

# instantiate the grid
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(GradientBoostingClassifier(random_state=0), parameters, cv=5)

start = time.time()
# fit the grid with data
grid.fit(train_data, train_labels)
end = time.time()

print("Gradient Boosting Training Time: ", end - start)

# view the complete results
# print("Grid Search Results:")
# print(grid.cv_results_)

print("Best Parameters: ", grid.best_params_)

# print the training accuracy, precision and recall
print("Training Accuracy for Gradient Boosting with best hyperparameters: ", grid.score(train_data, train_labels))

#print the precision and recall for training data
print("Training Precision for Gradient Boosting with best hyperparameters: ", precision_score(train_labels, grid.predict(train_data), average='macro'))

print("Training Recall for Gradient Boosting with best hyperparameters: ", recall_score(train_labels, grid.predict(train_data), average='macro'))

#print the confusion matrix for training data
from sklearn.metrics import confusion_matrix
print("Training Confusion Matrix for Gradient Boosting with best hyperparameters: ")
cm = confusion_matrix(train_labels, clf.predict(train_data))
print(cm)

# visualise the training confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# print the validation accuracy, precision and recall
print("Validation Accuracy for Gradient Boosting with best hyperparameters: ", grid.score(val_data, val_labels))

#print the precision and recall for validation data
# print("Validation Precision for Gradient Boosting with best hyperparameters: ", precision_score(val_labels, grid.predict(val_data), average='macro'))

# print("Validation Recall for Gradient Boosting with best hyperparameters: ", recall_score(val_labels, grid.predict(val_data), average='macro'))

# #print the confusion matrix for validation data
# from sklearn.metrics import confusion_matrix
# print("Validation Confusion Matrix for Gradient Boosting with best hyperparameters: ")
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


















# Extreme Gradient Boosting XGBoost



print("Extreme Gradient Boosting XGBoost: ")

# XGBoost Classifier
import xgboost as xgb
from xgboost import XGBClassifier





# train the xgboost classifier
xgb_clf = XGBClassifier()
xgb_clf.fit(train_data, train_labels)

# print the training and validation accuracy, precision and recall
from sklearn.metrics import accuracy_score, precision_score, recall_score

print("XGBoost Training Accuracy: ", accuracy_score(train_labels, xgb_clf.predict(train_data)))
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




print("XGBoost Validation Accuracy: ", accuracy_score(val_labels, xgb_clf.predict(val_data)))
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





















# perform a grid search on the selected features to find the best hyperparameters
from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators': [20, 30, 40, 50] , 'max_depth': [5, 6, 7, 8, 9, 10], 'subsample': [0.2, 0.3, 0.4, 0.5, 0.6]}

# instantiate the grid
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(XGBClassifier(random_state=0), parameters, cv=5)

start = time.time()

# fit the grid with data
grid.fit(train_data, train_labels)

end = time.time()

print("XGBoost Training Time: ", end - start)

# view the complete results
# print("Grid Search Results:")
# print(grid.cv_results_)

print("Best Parameters: ", grid.best_params_)

# print the training accuracy, precision and recall
print("Training Accuracy for XGBoost with best hyperparameters: ", grid.score(train_data, train_labels))

#print the precision and recall for training data
# print("Training Precision for XGBoost with best hyperparameters: ", precision_score(train_labels, grid.predict(train_data), average='macro'))

# print("Training Recall for XGBoost with best hyperparameters: ", recall_score(train_labels, grid.predict(train_data), average='macro'))

# #print the confusion matrix for training data
# from sklearn.metrics import confusion_matrix
# print("Training Confusion Matrix for XGBoost with best hyperparameters: ")
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

# print the validation accuracy, precision and recall
print("Validation Accuracy for XGBoost with best hyperparameters: ", grid.score(val_data, val_labels))

#print the precision and recall for validation data
# print("Validation Precision for XGBoost with best hyperparameters: ", precision_score(val_labels, grid.predict(val_data), average='macro'))

# print("Validation Recall for XGBoost with best hyperparameters: ", recall_score(val_labels, grid.predict(val_data), average='macro'))

# #print the confusion matrix for validation data
# from sklearn.metrics import confusion_matrix
# print("Validation Confusion Matrix for XGBoost with best hyperparameters: ")
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


















# Real time testing 

# Test your model with the image of your own face with different view angles and partially visible face. Report your accuracy with 10 such images.

# load the test images
import cv2
import os
import numpy as np

test_images = []
test_labels = []

# load the test images which are all face images so label them as 1
for filename in os.listdir('/content/drive/MyDrive/test'):
    img = cv2.imread(os.path.join('/content/drive/MyDrive/test',filename))
    if img is not None:
        
        # flatten the image
        img = cv2.resize(img, (32,32))
        img = img.flatten()
        print(img.shape)

        test_images.append(img)
        test_labels.append(1)


# convert the test images and labels to numpy arrays
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# print the shape of the test images and labels
print("Test Images Shape: ", test_images.shape)
print("Test Labels Shape: ", test_labels.shape)

  
# print the predictions for the test images
print("Predictions for the test images: ", xgb_clf.predict(test_images))


# print the test accuracy, precision and recall
print("Test Accuracy for XGBoost with best hyperparameters: ", xgb_clf.score(test_images, test_labels))