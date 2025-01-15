import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.cluster import contingency_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# loading the dataset
data = pd.read_csv('hotel_booking.csv', nrows=40000)

# display first few rows
# print(data.head())

# display structure of the dataset
# print(data.info())

# removing the irrelevant stuff
data = data.drop(['name', 'email', 'credit_card','phone-number', 'reservation_status_date',
              'reservation_status'], axis=1)

# check for missing values
# print(data.isnull().sum())

# Handle missing values (for example, filling with the mean for numerical and mode for categorical)
# data['some_numeric_column'].fillna(data['some_numeric_column'].mean(), inplace=True)
# data['some_categorical_column'].fillna(data['some_categorical_column'].mode()[0], inplace=True)

# IMPLEMENTATION
# encoding categorical variables using one-hot encoding
# avoiding the dummy variable trap (redundancy)
data = pd.get_dummies(data, drop_first=True)

# display the updated data frame
# print(data.head())

# splitting data into features and labels
X = data.drop('is_canceled', axis=1)  # features
y = data['is_canceled']  # target

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
# 15% of the data will be used for testing. The rest will be used for training

model = RandomForestClassifier(n_estimators=200, random_state=42)
# using 200 decision trees

# fitting the model on the training data
# i.e., training
model.fit(X_train, y_train)

# EVALUATION OF THE MODEL
# make predictions on the test set
y_pred = model.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'accuracy : {accuracy:.3f}')

# confusion matrix
# shows true positives, false positives, true negatives, and false negatives
da_conf_matrix = contingency_matrix(y_test, y_pred)
print('confusion matrix :')
print(da_conf_matrix)

# classification report
# precision, recall, F1-score, and support for each class in the target variable
da_class_report = classification_report(y_test, y_pred)
print('classification report :')
print(da_class_report)
