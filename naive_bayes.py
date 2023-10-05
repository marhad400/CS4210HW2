#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: Naive Bayes Classifier for weather data, outputting predicitons with at least 75% confidence
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#reading the training data in a csv file
#--> add your Python code here
db = []

with open('weather_training.csv', 'r') as csvfile:
	reader = csv.reader(csvfile)
	header = next(reader)
	for i, row in enumerate(reader):
		if i > 0:
			db.append(row)

num_rows = len(db)
num_cols = len(db[0])

X = [[0 for x in range(num_cols - 1)] for y in range(num_rows)]
Y = [0 for y in range(num_rows)]

feature_values_dict = {"Sunny": 1, "Overcast": 2, "Rain": 3, "Hot": 1, "Mild": 2, "Cool": 3, "Normal": 1, "High": 2, "Strong": 1, "Weak": 2}
class_dict = {"Yes": 1, "No": 2}

#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
for i in range(num_rows):
    for j in range(num_cols - 1): #excluding classification column
        key = db[i][j]
        if key in feature_values_dict:
            X[i][j] = feature_values_dict[key]

#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
Y = [0 for y in range(num_rows)]
for i in range(num_rows):
	key = db[i][-1]
	if key in class_dict:
		Y[i] = class_dict[key]

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
#--> add your Python code here
dbTest = []

with open('weather_test.csv', 'r') as csvfile:
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

#printing the header os the solution
#--> add your Python code here
for feature in header:
      print(feature, end=" ")
print()

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here

predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)
for i, probability in enumerate(probabilities):
      confidence = probability[0]
      if confidence >= 0.75:
        for j in range(len(X_test[0])):
                print(dbTest[i][j], end=" ")
        print("%.2f" % confidence)