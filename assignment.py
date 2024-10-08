import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report




heart_data = pd.read_csv('heart_disease_data.csv')
# print(heart_data.head())
# print(heart_data.shape)

#checking for missing values

# print(heart_data.isnull().sum())

#statistical measures about the data
# print(heart_data.describe())

#checking the distribution of target variable  (165 have heart disease 1 represents defective and 0 has predictive heart)
# print(heart_data['target'].value_counts())

#spliting features and target
x=heart_data.drop(columns='target',axis=1)
y=heart_data['target']
# print(x)
# print(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)
print(x.shape,x_train.shape,x_test.shape)

#model training  i am using logistic regression because it is useful for binary data
model =LogisticRegression()

#training the LogisticRegression model with training data
model.fit(x_train,y_train)

#model evaluation
#accuracy prediction on test data
x_train_prediction = model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print("accuracy on training data:", training_data_accuracy)

#accurancy prediction on train data
x_test_prediction = model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print("accuracy on test data:", test_data_accuracy)

#there is no overfitting of data since there is no huge gap in training in test and train date

#building a prediction system
input_data=[46,1,0,120,249,0,0,144,0,0.8,2,0,3]
#change to input array to numpy_array
numpy_array=np.asarray(input_data)
 
#reshape the numpy array as we are predicting for only on instance
input_data_reshape = numpy_array.reshape(1,-1)
prediction= model.predict(input_data_reshape)
print(prediction)



if(prediction[0]==0):
    print("the person doesnt have a heart disease")
else:
    print("the person has heart disease")

