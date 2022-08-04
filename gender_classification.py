import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#getting and storing Dataframe into df:
df = pd.read_csv('gender_classification.csv')

#we drop the target output column "gender" then store the Dataframe into x:
x = df.drop(['gender'],axis=1).values

#we take the target output column "gender" and store it into y:
y = df['gender'].values

#because we want the model to predict either male or female, we use the DecisionTreeClassifier function from sklearn:
model = DecisionTreeClassifier()

#we split our Dataframe using train_test_split function (70% training, 30% testing):
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

#model training:
model.fit(x_train,y_train)

#Testing model by making predictions:
predictions = model.predict(x_test)

#calculating the accuracy (with accuracy_score function from sklearn) to see how good the model's predictions are:
score = accuracy_score(y_test, predictions)
