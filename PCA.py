import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('wine.csv')
df.head()
df.describe()
print(df)

y = df['Wine']           # Select the column with label 'Wine' as the target variable
X = df.drop('Wine', axis=1)  # Drop the column with label 'Wine' to create the features

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
#Importing scaler
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = pd.DataFrame(X_train)
X_test= pd.DataFrame(X_test)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)
from sklearn.metrics import classification_report
print("The classification report is:{}".format(classification_report(y_test, clf.predict(X_test))))

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_train = pca.fit_transform(X_train)
pca_test = pca.transform(X_test)

pc_model = LogisticRegression()
pc_model.fit(pca_train, y_train)
pc_model
print("The classification report is:{}".format(classification_report(y_test,pc_model.predict(pca_test))))