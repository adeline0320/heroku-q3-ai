import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score,recall_score,f1_score
from sklearn.metrics import roc_auc_score
from sklearn import svm, tree
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import RandomOverSampler



st.title("Question 3")

file = pd.read_csv("Bank_CreditScoring.csv")

mean_loan = file["Loan_Amount"].mean()
st.write("The mean of loan amount is ",mean_loan)

side_income = file["Number_of_Side_Income"].max()
st.write("The maximum side income is: ",side_income)

median_score = file["Score"].median()
st.write("The median of the score is",median_score)

decision = file["Decision"].value_counts().sort_index()
st.bar_chart(data = decision)

employ = file["Employment_Type"].value_counts().sort_index()

st.bar_chart(data = employ)

fig, ax = plt.subplots()
sns.heatmap(file.corr(),cmap="cividis")
st.write(fig)


st.header("Clustering")
st.header("Dumification")

X = file.drop('Score',axis = 1)
y = file['Score']

X = pd.get_dummies(X,drop_first = True)


option = st.number_input('Number',step = 1)
st.write(option)
if option == '2':
    st.write("You have select:",option)

if option != '<select>':
    st.write('here')
    newData =pd.read_csv('K_mean_' + str(option)+'.csv')
    st.write("Check again the class variable:")
    decision = file["Decision"].value_counts().sort_index()
    st.bar_chart(data = decision)

    st.write("Oversample")
    oversample = RandomOverSampler(sampling_strategy = 'minority')
    X = newData.drop('Decision',axis = 1)
    y = newData['Decision']
    X = pd.get_dummies(X, drop_first=True) 
    X_over,y_over = oversample.fit_resample(X,y)
    X_train,X_test,y_train,y_test = train_test_split(X_over,y_over,test_size = 0.3,random_state = 42)

    clf=RandomForestClassifier()
    #n_estimators=800, criterion='gini', max_depth=None,min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False,class_weight=None
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    prob_y_2 = clf.predict_proba(X_test)
    prob_y_2 = [p[0] for p in prob_y_2]
    st.write('Recall:',recall_score(y_test, y_pred,pos_label = "Accept"))
    st.write('Accuracy score:',accuracy_score(y_test, y_pred))
    st.write('F1 score:',f1_score(y_test, y_pred,pos_label = "Accept"))
    st.write('roc_auc_score',roc_auc_score(y_test,prob_y_2 ) )

    conmat = confusion_matrix(y_test, y_pred)
    val = np.mat(conmat) 
    classnames = list(set(y_train))
    df_cm = pd.DataFrame(
        val, index=classnames, columns=classnames, 
    )
    fig, ax = plt.subplots()
    heatmap = sns.heatmap(df_cm, annot=True, cmap="Blues",ax = ax)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Results')
    st.write(fig)

    nb = GaussianNB()
    nb.fit(X_train,y_train)
    y_predGB = nb.predict(X_test)
    prob_y_3 = nb.predict_proba(X_test)
    prob_y_3 = [p[0] for p in prob_y_3]
    #print('ROCAUC score:',roc_auc_score(y_test, y_pred))
    st.write('Accuracy score NB Decision:',accuracy_score(y_test, y_predGB))
    #st.write('roc_auc_score',roc_auc_score(y_test,prob_y_3) )
    conmat = confusion_matrix(y_test, y_predGB)
    val = np.mat(conmat) 
    classnames = list(set(y_train))
    df_cm = pd.DataFrame(
        val, index=classnames, columns=classnames, 
    )
    fig, ax = plt.subplots()
    heatmap = sns.heatmap(df_cm, annot=True, cmap="Blues",ax = ax)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Results')

    st.write(fig)

    st.header("Classification Problem 2")
    newFile =pd.read_csv('K_mean_' + str(option)+'.csv')
    label = newFile["Label"].value_counts().sort_index()
    plt.xlabel("Label")
    plt.ylabel("Frequency")
    st.bar_chart(data = label)

    X = newData.drop('Label',axis = 1)
    y = newData['Label'].astype(str)
    
    X = pd.get_dummies(X, drop_first=True) 
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 42)

    clf=RandomForestClassifier()
    #n_estimators=800, criterion='gini', max_depth=None,min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False,class_weight=None
    clf.fit(X_train,y_train)
    y_predC2=clf.predict(X_test)
    prob_y_4 = clf.predict_proba(X_test)
    prob_y_4 = [p[1] for p in prob_y_4]
    st.write('Accuracy score RF Label:',accuracy_score(y_test, y_predC2))
    #st.write('roc_auc_score',roc_auc_score(y_test,prob_y_4 ))
    conmat = confusion_matrix(y_test, y_predC2)
    val = np.mat(conmat) 
    classnames = list(set(y_train))
    df_cm = pd.DataFrame(
        val, index=classnames, columns=classnames, 
    )
    fig, ax = plt.subplots()
    heatmap = sns.heatmap(df_cm, annot=True, cmap="Blues",ax = ax)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Results')
    
    st.write(fig)

    nb = GaussianNB()
    nb.fit(X_train,y_train)
    y_predGB = nb.predict(X_test)
    prob_y_3 = nb.predict_proba(X_test)
    prob_y_3 = [p[1] for p in prob_y_3]
    #print('ROCAUC score:',roc_auc_score(y_test, y_pred))
    #print('roc_auc_score',roc_auc_score(y_test,prob_y_3) )
    conmat = confusion_matrix(y_test, y_predGB)
    val = np.mat(conmat) 
    classnames = list(set(y_train))
    df_cm = pd.DataFrame(
        val, index=classnames, columns=classnames, 
    )
    fig, ax = plt.subplots()
    heatmap = sns.heatmap(df_cm, annot=True, cmap="Blues",ax = ax)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Results')
    
    st.write(fig)
    st.write('Accuracy score NB Label:',accuracy_score(y_test, y_predGB))