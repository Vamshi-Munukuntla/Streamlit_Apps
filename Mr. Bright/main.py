import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Set Title
st.title('ML Automation App')

# set subtitle
st.write("""
    ## A simple data app with Streamlit
""")

st.write("""
    ### Let's Explore different classifiers and datasets
""")

dataset_name = st.sidebar.selectbox('Select Dataset', ('Breast Cancer', 'Wine', 'Iris'))

classifier_name = st.sidebar.selectbox('Select Classifier', ('SVM', 'KNN'))


def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    x = data.data
    y = data.target
    feature_names = data.feature_names
    target_name = data.target_names

    return x, y, feature_names, target_name


x, y, x_columns, y_columns = get_dataset(dataset_name)
X = pd.DataFrame(x, columns=x_columns)
# Y = pd.DataFrame(y, columns=y_columns)
st.dataframe(X)
st.success('Data is successfully uploaded')
st.write('Shape of your dataset:', x.shape)
st.write('Number of Unique target variables: ', len(np.unique(y)))
st.write('Unique target variables:', np.unique(y))
st.write('Target Variable Names:', y_columns)

fig, ax = plt.subplots()
sns.boxplot(data=x)
st.pyplot(fig)


def add_parameter(name_of_clf):
    params = dict()
    if name_of_clf == "SVM":
        c = st.sidebar.slider('C', 0.01, 15.0)
        params['C'] = c
    elif name_of_clf == "KNN":
        k = st.sidebar.slider("K", 1, 15, step=2)
        params['K'] = k
    return params


params = add_parameter(classifier_name)


# Accessing our classifier
def get_classifier(name_of_clf, params):
    clf = None
    if name_of_clf == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    return clf


clf = get_classifier(classifier_name, params)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=20)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(y_pred)
st.write(f"Accuracy Score for {classifier_name}: {round(accuracy, 3)}")




