
import math, csv
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn import neighbors, svm, tree, metrics
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import *

#  reading dataset
filepath = 'Project/src/heart.csv'
dataset = pd.read_csv(filepath)



print(dataset.describe())
#Gives number of data points and features


#Data types of features
print(dataset.dtypes)

#Correlation of features
print(dataset.corr())


#Counting Data Points by Chest Pain Type, Age and Sex
print('\nNumber of Chest Pain types: \n', dataset.cp.value_counts())
print('\nNumber of Incidents by Age: \n', dataset.age.value_counts())
print('\nNumber of Incidents by Sex: \n', dataset.sex.value_counts())


#Sex Histogram
plt.hist(dataset.sex, bins=2)
plt.xlabel('0 - Female, 1 - Male')
plt.ylabel('Frequency')
plt.title('Sex of Patients')
plt.show()


#Age Histogram
plt.hist(dataset.age, bins=12)
plt.xlabel('Age (Years)')
plt.ylabel('Frequency')
plt.title('Age of Patients')
plt.show()

#Chol Histogram
plt.hist(dataset.chol, bins=10)
plt.xlabel('Cholesterol in mg/dl')
plt.ylabel('Frequency')
plt.title('Patients by Cholesterol Level')
plt.show()

#Blood Pressure Histogram
plt.hist(dataset.trtbps, bins=12)
plt.xlabel('Pressure (mm Hg)')
plt.ylabel('Frequency')
plt.title('Patients by Blood Pressure')
plt.show()

#Heart Rate Histogram
plt.hist(dataset.thalachh, bins=12)
plt.xlabel('Beats per minute')
plt.ylabel('Frequency')
plt.title('Patients by Max Heart Rate')
plt.show()

#Chest pain Histogram
plt.hist(dataset.cp, bins=4)
plt.xlabel('Type (0-3)')
plt.ylabel('Frequency')
plt.title('Patients by Chest Pain Type')
plt.show()



#Correlation Matrix (Heatmap)
plt.figure(figsize=(9, 6))
sb.heatmap(dataset.corr(), annot=True, cmap="inferno")
plt.title("Correlation Matrix for Features")
plt.show()


#-------------------Classification------------------------
#data separation
x = dataset.drop(['output'], axis = 1)
y = dataset[['output']]

# splitting into train and test
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

#KNN
knn_results = []
knn_matrices = []
length = len(x_train)
for i in range(len(x_train)):
    n_neighbors = i+1
    nn = neighbors.KNeighborsClassifier(n_neighbors) 
    nn.fit(x_train, y_train)
    y_predicted = nn.predict(x_test)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_predicted).ravel()
    accuracy = round(((tp + tn) / (tp + tn + fp + fn)), 3)
    precision_c1 = round((tp / (tp + fp)),3)
    recall_c1 = round((tp / (tp + fn)),3)
    precision_c2 = round((tn / (tn + fn)),3)
    recall_c2 = round((tn / (tn + fp)),3)
    matrix = [accuracy, precision_c1, precision_c2, recall_c1, recall_c2]
    
    knn_results.append([n_neighbors, round(accuracy_score(y_test, y_predicted),3)])
    knn_matrices.append([n_neighbors,matrix])

# results
print('KNN')  
max = knn_results[0][1]
k_max = 0
matrix = []
for i in range(len(knn_results)):
    if max < knn_results[i][1]:
        max = knn_results[i][1]
        k_max = i + 1
        matrix = knn_matrices[i][1]

print(' K: ',k_max)
print(' accuracy:', matrix[0],"\n",
      'precision (low-risk)',matrix[1],"\n",
      'recall (low-risk)',matrix[2],"\n",
      'precision (high-risk)',matrix[3],"\n",
      'recall (high-risk)',matrix[4],)

#Decision-Tree
dt = DecisionTreeClassifier(random_state = 7,criterion='entropy')
dt.fit(x_train, y_train)
y_predicted = dt.predict(x_test)
'''

'''
img = plt.figure(figsize=(70,40), dpi = 100)
_ = tree.plot_tree(dt, feature_names=['age','sex','cp','trtbps',
                                      'chol','fbs','restecg','thalachh',
                                      'exng','oldpeak','slp','caa','thall'],  
                   class_names=['low-risk','high-risk'],
                   filled=True)
img.savefig("decisionTree.png")
'''
'''
tn, fp, fn, tp = confusion_matrix(y_test, y_predicted).ravel()
accuracy = round(((tp + tn) / (tp + tn + fp + fn)), 3)
precision_c1 = round((tp / (tp + fp)),3)
recall_c1 = round((tp / (tp + fn)),3)
precision_c2 = round((tn / (tn + fn)),3)
recall_c2 = round((tn / (tn + fp)),3)
matrix = [accuracy, precision_c1, precision_c2, recall_c1, recall_c2]
print('\n')
print('Decision tree')
print(' accuracy:', matrix[0],"\n",
      'precision (low-risk)',matrix[1],"\n",
      'recall (low-risk)',matrix[2],"\n",
      'precision (high-risk)',matrix[3],"\n",
      'recall (high-risk)',matrix[4],)

# random forest
rf = RandomForestClassifier(random_state=7, criterion='entropy')
rf.fit(x_train, y_train)
y_predicted = rf.predict(x_test)

tree_from_forest = rf.estimators_[5]
img = plt.figure(figsize=(70,40), dpi = 100)
_ = tree.plot_tree(tree_from_forest, feature_names=['age','sex','cp','trtbps',
                                      'chol','fbs','restecg','thalachh',
                                      'exng','oldpeak','slp','caa','thall'],  
                   class_names=['low-risk','high-risk'],
                   filled=True)
img.savefig("randomForest.png")

tn, fp, fn, tp = confusion_matrix(y_test, y_predicted).ravel()
accuracy = round(((tp + tn) / (tp + tn + fp + fn)), 3)
precision_c1 = round((tp / (tp + fp)),3)
recall_c1 = round((tp / (tp + fn)),3)
precision_c2 = round((tn / (tn + fn)),3)
recall_c2 = round((tn / (tn + fp)),3)
matrix = [accuracy, precision_c1, precision_c2, recall_c1, recall_c2]
print('\n')
print('Random Forest')
print(' accuracy:', matrix[0],"\n",
      'precision (low-risk)',matrix[1],"\n",
      'recall (low-risk)',matrix[2],"\n",
      'precision (high-risk)',matrix[3],"\n",
      'recall (high-risk)',matrix[4],)

#ANN
ANN = MLPClassifier()
ANN.fit(x_train, y_train)
y_predicted = ANN.predict(x_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_predicted).ravel()
accuracy = round(((tp + tn) / (tp + tn + fp + fn)), 3)
precision_c1 = round((tp / (tp + fp)),3)
recall_c1 = round((tp / (tp + fn)),3)
precision_c2 = round((tn / (tn + fn)),3)
recall_c2 = round((tn / (tn + fp)),3)
matrix = [accuracy, precision_c1, precision_c2, recall_c1, recall_c2]
print('\n')
print('ANN')
print(' accuracy:', matrix[0],"\n",
      'precision (low-risk)',matrix[1],"\n",
      'recall (low-risk)',matrix[2],"\n",
      'precision (high-risk)',matrix[3],"\n",
      'recall (high-risk)',matrix[4],)
