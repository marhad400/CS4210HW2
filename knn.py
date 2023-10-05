#-------------------------------------------------------------------------
# AUTHOR: Mark Haddad
# FILENAME: knn.py
# SPECIFICATION: KNN algorithm on caresian point dataset, using predictions to find an error rate
# FOR: CS 4210- Assignment #2
# TIME SPENT: 45 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv
# import warnings filter
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

db = []

dbTest = []

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         dbTest.append (row)

#loop your data to allow each instance to be your test set
num_rows = len(db)
num_cols = len(db[0])

num_rows_test = len(dbTest)
num_cols_test = len(dbTest[0])

X = [[0 for x in range(num_cols - 1)] for y in range(num_rows)]
Y = [0 for y in range(num_rows)]

X_test = [[0 for x in range(num_cols - 1)] for y in range(num_rows)]

class_dict = {"+": 1, "-": 2}

# adding test features
for i in range(num_rows_test):
    for j in range(num_cols_test - 1): #excluding classification column
        X_test[i][j] = float(dbTest[i][j])

for i in range(num_rows):

    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]. Convert each feature value to
    # float to avoid warning messages
    #--> add your Python code here
    for j in range(num_cols - 1):
       X[i][j] = float(db[i][j])
    
    training_instance = X[i]
    print("Sample:", training_instance)

    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]. Convert each
    #  feature value to float to avoid warning messages
    #--> add your Python code here
    key = db[i][-1]
    if key in class_dict:
        Y[i] = float(class_dict[key])
        training_class = Y[i]
    print("Class Label:", training_class)

    #store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testSample = X_test[i]

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

#use your test sample in this iteration to make the class prediction. For instance:
#class_predicted = clf.predict([[1, 2]])[0]
#--> add your Python code here
predictions = clf.predict(X_test)
print("Predictions:", predictions)

#compare the prediction with the true label of the test instance to start calculating the error rate.
#--> add your Python code here
wrong_predictions = 0
total_predictions = len(predictions)
for i in range(total_predictions):
    if int(predictions[i]) != Y[i]:
       wrong_predictions += 1
#print the error rate
#--> add your Python code here
error_rate = wrong_predictions / total_predictions
print("Error Rate:", error_rate)