# from sklearn.ensemble import RandomForestClassifier
# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# #from kneed import KneeLocator
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.cluster import KMeans
# import seaborn as sns 
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score, cross_val_predict
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import metrics
# from sklearn import svm, tree

# st.set_page_config(layout = 'wide')

# st.title("Question 3")

# file = pd.read_csv("Bank_CreditScoring.csv")
# mean_loan = file["Loan_Amount"].mean()
# st.text(mean_loan)
# side_income = file["Number_of_Side_Income"].max()
# st.text(side_income)
# median_score = file["Score"].median()
# st.text(median_score)

# employ = file["Employment_Type"].value_counts().sort_index()
# # plt.xlabel("Type of employment")
# # plt.ylabel("Frequency")
# # plt.title("Frequency of type of employment")
# st.bar_chart(data = employ)

# fig, ax = plt.subplots()
# sns.heatmap(file.corr(),cmap="cividis")
# st.write(fig)

# X = file.drop('Score',axis = 1)
# y = file['Score']



# st.header("Clustering")
# st.header("Dumification")
# X = pd.get_dummies(X,drop_first = True)
# st.table(X.head())

# distortions = []#initial
# for i in range(2,11):
#     km = KMeans(
#         n_clusters = i, init = 'random',
#         n_init = 10,max_iter = 300,
#         tol = 1e-04,random_state = 0
#     )
#     km.fit(X)
#     distortions.append(km.inertia_)
# # plt.plot(range(2,11),distortions,marker = 'o')
# # plt.xlabel('Number of clusters')
# # plt.ylabel('Distortion')
# # plt.show()

# KM = KMeans(n_clusters = 5,random_state = 12)
# KM.fit(X)
# label = KM.predict(X)

# file['Label']=label
# # print(file['Score'])
# # print(file)
# # sns.scatterplot(X['Monthly_Salary'], X['Loan_Amount'], hue=file['Score'])


# X = file.drop('Decision',axis = 1)
# y = file['Decision']
# X = X.astype(str)

# X = pd.get_dummies(X,drop_first = True)
# st.header("Training data")
# st.table(X.head())
# st.header("Test data")
# st.table(y.head())

# from sklearn.naive_bayes import GaussianNB
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)

# # st.header("Classification model 1")
# # st.text("Gaussian Naive Bayer")
# # nb = GaussianNB()
# # nb.fit(X_train,y_train)
# # y_pred = nb.predict(X_test)
# # nb.score(X_test, y_test)

# # st.text(nb.score(X_test, y_test))



# # clf.score(X_test, y_test)
# st.header("Classification model 1")
# st.text("Decision Tree")
# X = file.drop('',axis = 1)
# y = file['Decision']
# X = X.astype(str)
# st.text("Accuracy:")
# st.text(metrics.accuracy_score(y_test,y_predCLF))

# st.text("SVM")
# svm = svm.SVC()
# svm = svm.fit(X_train,y_train)
# y_predSVM = svm.predict(X_test)
# st.table(y_predSVM)
# acc = accuracy_score(y_test, y_predSVM)
# st.text("Accuracy of %s is %s"%("SVM", acc))
# cm = confusion_matrix(y_test, y_predSVM)
# st.text("Confusion Matrix of %s is %s"%("SVM", cm))

# st.text("Random Forest")
# rf = RandomForestClassifier()
# rf = rf.fit(X_train,y_train)
# y_predrf = rf.predict(X_test)
# st.table(y_predrf)
# acc = accuracy_score(y_test, y_predrf)
# st.text("Accuracy of %s is %s"%("RF", acc))
# cm = confusion_matrix(y_test, y_predrf)
# st.text("Confusion Matrix of %s is %s"%("RF", cm))