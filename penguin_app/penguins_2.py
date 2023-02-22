import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Palmer's Penguins")
st.markdown('Use this Streamlit app to make your own scatter plot about penguins!')


penguin_file = st.file_uploader('Select Your Local Penguins CSV (default provided)')

if penguin_file is not None:
    penguins_df = pd.read_csv(penguin_file)
else:
    st.stop()

variables = ['bill_length_mm', 'bill_depth_mm',
             'flipper_length_mm', 'body_mass_g']

selected_x_var = st.selectbox('What do want the x variable to be?', variables)
selected_y_var = st.selectbox('What about the y?', variables)

sns.set_style('darkgrid')
sns.set_style('darkgrid')
markers = {'Adelie': 'X', 'Gentoo': 's', 'Chinstrap': 'o'}
fig, ax = plt.subplots()
ax = sns.scatterplot(data=penguins_df, x=selected_x_var,
                     y=selected_y_var, hue='species', markers=markers,
                     style='species')
plt.xlabel(selected_x_var)
plt.ylabel(selected_y_var)
plt.title(f"Scatter-plot of Palmer's Penguins")
plt.tight_layout()
st.pyplot(fig)
