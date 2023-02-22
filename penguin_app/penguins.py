import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

st.title("Palmer's Penguins")
st.markdown('Use this Streamlit app to make your own scatter plot about penguins!')

# import our data
penguin_file = st.file_uploader('Select Your Local Penguins CSV (default provided)')


@st.cache_data()
def load_file(file):
    time.sleep(3)
    if file is not None:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv(r"D:\Streamlit\penguin_app\penguins.csv")
    return df


penguins_df = load_file(penguin_file)

variables = ['bill_length_mm', 'bill_depth_mm',
             'flipper_length_mm', 'body_mass_g']

selected_x_var = st.selectbox('What do want the x variable to be?', variables)

selected_y_var = st.selectbox('What about the y?', variables)

selected_gender = st.selectbox('What gender do you want to filter for?',
                               ['all penguins', 'male penguins', 'female penguins'])

if selected_gender == 'male penguins':
    penguins_df = penguins_df.query('sex == "male"')
elif selected_gender == 'female penguins':
    penguins_df = penguins_df.query('sex == "female"')
else:
    pass

sns.set_style('darkgrid')
markers = {'Adelie': 'X', 'Gentoo': 's', 'Chinstrap': 'o'}
fig, ax = plt.subplots()
ax = sns.scatterplot(data=penguins_df, x=selected_x_var,
                     y=selected_y_var, hue='species', markers=markers,
                     style='species')
plt.xlabel(selected_x_var)
plt.ylabel(selected_y_var)
plt.title(f"Scatter-plot of Palmer's Penguins: {selected_gender}")
plt.tight_layout()
st.pyplot(fig)
