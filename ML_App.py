import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.metrics import accuracy_score

matplotlib.use('Agg')
st.title('ML APP')


def main():
    activities = ['EDA', 'Visualisation', 'Model Building', 'About Us']
    option = st.sidebar.radio('Select an option:', activities)

    # DEALING WITH THE EDA PART

    if option == 'EDA':
        st.subheader('Exploratory Data Analysis')

        data = st.file_uploader('Upload dataset: ', type=['csv', 'xlsx', 'txt', 'json'])

        if data is not None:
            st.success("Data Successfully Uploaded.")
            df = pd.read_csv(data)
            st.write('Top 10 rows in the data')
            st.dataframe(df.head(10))

            if st.checkbox('Display Shape'):
                st.write(df.shape)
            if st.checkbox('Display Columns'):
                st.write(df.columns)
            if st.checkbox('Select Multiple options'):
                selected_columns = st.multiselect('Select Preferred Columns', df.columns)
                df1 = df[selected_columns]
                st.dataframe(df1)

            if st.checkbox("Display summary "):
                st.write(df.describe().T)

            if st.checkbox('Display null values'):
                st.write(df.isnull().sum())

            if st.checkbox('Display data types'):
                st.write(df.dtypes)

            if st.checkbox('Display Correlation'):
                st.write(df.corr())

    # DEALING WITH VISUALISATION PART
    elif option == 'Visualisation':
        st.subheader('Visualisation')
        data = st.file_uploader('Upload dataset: ', type=['csv', 'xlsx', 'txt', 'json'])

        if data is not None:
            st.success("Data Successfully Uploaded.")
            df = pd.read_csv(data)

            if st.checkbox('Selected Multiple columns to plot'):
                selected_columns = st.multiselect('Select your preferred columns:', df.columns)
                df1 = df[selected_columns]
                st.dataframe(df1)

            if st.checkbox('Display Heatmap'):
                fig, ax = plt.subplots()
                corr = df.corr()
                mask = np.triu(np.ones_like(corr, dtype=bool))
                cmap = sns.diverging_palette(230, 20, as_cmap=True)
                sns.heatmap(corr, mask=mask, cmap='viridis', vmax=1, center=0, annot=True,
                            square=True, linewidths=.5, cbar_kws={"shrink": .5})
                st.pyplot(fig)

            # if st.checkbox('Display Pair plot'):
            #     fig, ax = plt.subplots()
            #     st.write(sns.pairplot(data=df, diag_kind='kde'))
            #     st.pyplot()

            if st.checkbox('Display Pie Plot'):
                fig, ax = plt.subplots()
                all_columns = df.columns.to_list()
                pie_columns = st.selectbox('select column to display', all_columns)
                pie_chart = df[pie_columns].value_counts().plot.pie(autopct="%1.1f%%")
                st.pyplot(fig)

    elif option == 'Model Building':
        st.subheader('Model Building')
        data = st.file_uploader('Upload dataset: ', type=['csv', 'xlsx', 'txt', 'json'])

        if data is not None:
            st.success("Data Successfully Uploaded.")
            df = pd.read_csv(data)

            if st.checkbox('Select Multiple Columns'):
                new_data = st.multiselect('Select your preferred columns', df.columns)
                df1 = df[new_data]
                st.dataframe(df1)

                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]

                seed = st.sidebar.slider('Seed', 1, 200)
                models = ('KNN', 'SVM', 'Logistic Regression', 'Naive Bayes', 'Decision Tree')
                classifier_name = st.sidebar.selectbox('Select a classifier for modelling', models)

            def add_parameter(name_of_classifier):
                params = dict()
                if name_of_classifier == 'SVM':
                    C = st.sidebar.slider('C', 0.01, 15.0)
                    params['C'] = C
                elif name_of_classifier == 'KNN':
                    K = st.sidebar.slider('K', min_value=1, max_value=15, step=2)
                    params['K'] = K
                return params

            # calling the function
            model_parameters = add_parameter(classifier_name)

            # Defining a function for classifier
            def get_classifier(name_of_classifier, params):
                clf = None
                if name_of_classifier == 'SVM':
                    clf = SVC(C=params['C'])
                elif name_of_classifier == 'KNN':
                    clf = KNeighborsClassifier(n_neighbors=params['K'])
                elif name_of_classifier == 'Logistic Regression':
                    clf = LogisticRegression()
                elif name_of_classifier == 'Naive Bayes':
                    clf = GaussianNB()
                elif name_of_classifier == 'Decision Tree':
                    clf = DecisionTreeClassifier()
                else:
                    st.warning('Select your choice of algorithm for model building')
                    return clf

            get_classifier(classifier_name, params)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)


if __name__ == "__main__":
    main()
