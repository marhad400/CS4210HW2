#-------------------------------------------------------------------------
# AUTHOR: Mark Haddad
# FILENAME: decision_tree_2.py
# SPECIFICATION: Decision Tree on three different datasets, using each to find an average prediction accuracy
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

dbTest = []

feature_values_dict = {"Young": 1, "Prepresbyopic": 2, "Presbyopic": 3, "Myope": 1, "Hypermetrope": 2, "Yes": 1, "No": 2, "Normal": 1, "Reduced": 2}
class_dict = {"Yes": 1, "No": 2}

with open('contact_lens_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:
            dbTest.append(row)

num_rows_test = len(dbTest)
num_cols_test = len(dbTest[0])

X_test = [[0 for x in range(num_cols_test - 1)] for y in range(num_rows_test)]
for i in range(num_rows_test):
    for j in range(num_cols_test - 1): #excluding classification column
        key = dbTest[i][j]
        if key in feature_values_dict:
            X_test[i][j] = feature_values_dict[key]

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    num_rows = len(dbTraining)
    num_cols = len(dbTraining[0])
    
    X = [[0 for x in range(num_cols - 1)] for y in range(num_rows)]
    for i in range(num_rows):
        for j in range(num_cols - 1): #excluding classification column
            key = dbTraining[i][j]
            if key in feature_values_dict:
                X[i][j] = feature_values_dict[key]

    #transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here

    Y = [0 for y in range(num_rows)]
    for i in range(num_rows):
        key = dbTraining[i][-1]
        if key in class_dict:
            Y[i] = class_dict[key]

    #loop your training and test tasks 10 times here
    accuracy_sum = 0
    for i in range (10):

       #fitting the decision tree to the data setting max_depth=3
       clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
       clf = clf.fit(X, Y)

       #read the test data and add this data to dbTest
       #--> add your Python code here
       predictions = clf.predict(X_test)
       print("Predictions:", predictions)
       #transform the features of the test instances to numbers following the same strategy done during training,
       #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
       #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
       #--> add your Python code here

       #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
       #--> add your Python code here
       correct_predictions = 0
       total_predictions = len(predictions)
       for j in range(total_predictions):
           if predictions[j] == Y[j]:
               correct_predictions += 1
       accuracy = correct_predictions / total_predictions
       accuracy_sum += accuracy
       print("Accuracy for prediction:", accuracy)
       print()
    #find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here
    dataset_average = accuracy_sum / 10

    #print the average accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print(f"Final accuracy when training on {ds}:", dataset_average)
    print()