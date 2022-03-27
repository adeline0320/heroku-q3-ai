import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

st.set_page_config(layout = 'wide')

st.title("Question 3")
st.header("Clustering")
file = pd.read_csv("Bank_CreditScoring.csv")
X = file.drop('Score',axis = 1)
y = file['Score']

st.header("Dumification")
X = pd.get_dummies(X,drop_first = True)
st.table(X.head())

distortions = []#initial
for i in range(2,11):
    km = KMeans(
        n_clusters = i, init = 'random',
        n_init = 10,max_iter = 300,
        tol = 1e-04,random_state = 0
    )
    km.fit(X)
    distortions.append(km.inertia_)
plt.plot(range(2,11),distortions,marker = 'o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

KM = KMeans(n_clusters = 5,random_state = 12)
KM.fit(X)
label = KM.predict(X)

file['Label']=label
# print(file['Score'])
# print(file)
# sns.scatterplot(X['Monthly_Salary'], X['Loan_Amount'], hue=file['Score'])


X = file.drop('Score',axis = 1)
y = file['Score'].apply(str)

X = pd.get_dummies(X,drop_first = True)
st.header("Training data")
st.table(X.head())
st.header("Test data")
st.table(y.head())
from sklearn.naive_bayes import GaussianNB
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)

nb = GaussianNB()
nb.fit(X_train,y_train)

y_pred = nb.predict(X_test)


nb.score(X_test, y_test)
st.header("Classification model 1")
st.text("Gaussian Naive Bayer")
st.text(nb.score(X_test, y_test))

clf = DecisionTreeClassifier()
clf.get_params()

clf = clf.fit(X_train,y_train)

y_predCLF = clf.predict(X_test)

clf.score(X_test, y_test)
st.header("Classification model 2")
st.text("Decision Tree")
st.text("Accuracy:")
st.text(metrics.accuracy_score(y_test,y_predCLF))