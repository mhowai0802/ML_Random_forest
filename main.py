import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("other_files/images_analyzed_productivity1.csv")

df.drop(['Images_Analyzed'], axis=1, inplace=True)
df.drop(['User'], axis=1, inplace=True)
df.Productivity[df.Productivity == 'Good'] = 1
df.Productivity[df.Productivity == 'Bad'] = 2
print(df.head())
Y = df["Productivity"].values
Y=Y.astype('int')
X = df.drop(labels = ["Productivity"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=20)

model = RandomForestClassifier(n_estimators = 10, random_state = 30)
model.fit(X_train, y_train)
prediction_test = model.predict(X_test)
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))

feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp)