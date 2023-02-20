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
    activities = ['EDA', 'Visualisation', 'model', 'About Us']
    option = st.sidebar.selectbox('Select an option:', activities)

    # DEALING WITH THE EDA PART

    if option == 'EDA':
        st.subheader('Exploratory Data Analysis')

        data = st.file_uploader('Upload dataset: ', type=['csv', 'xlsx', 'txt', 'json'])
        st.success("Data Successfully Uploaded.")
        if data is not None:
            df = pd.read_csv(data)
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






    # elif option == 'Visualisation':
    #     pass
    # elif option == 'model':
    #     pass
    # else:
    #     pass


if __name__ == "__main__":
    main()