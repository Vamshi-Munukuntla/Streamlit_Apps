import streamlit as st
from streamlit_lottie import st_lottie
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


height = st.sidebar.slider('Height', value=400, min_value=200, max_value=1200, step=100)
width = st.sidebar.slider('Width', value=600, min_value=200, max_value=1200, step=100)
speed = st.sidebar.slider('Speed', value=1.0, min_value=0.5, max_value=4.0, step=0.2)

lottie_penguin = load_lottie_url('https://assets9.lottiefiles.com/private_files/lf30_lntyk83o.json')
st_lottie(lottie_penguin, height=height, speed=speed, width=width)

st.title("Palmer's Penguins")
st.markdown('Use this Streamlit app to make your own scatterplot about penguins!')

variables = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
selected_x_var = st.selectbox('What do want the x variable to be?', variables)
selected_y_var = st.selectbox('What about the y?', variables)

penguin_file = st.file_uploader('Select Your Local Penguins CSV')

if penguin_file is not None:
    penguins_df = pd.read_csv(penguin_file)
else:
    penguins_df = pd.read_csv('penguins.csv')

sns.set_style('darkgrid')
markers = {"Adelie": "X", "Gentoo": "s", "Chinstrap": 'o'}

fig, ax = plt.subplots()
sns.scatterplot(data=penguins_df, x=selected_x_var, y=selected_y_var,
                hue='species', markers=markers, style='species')
plt.xlabel(selected_x_var)
plt.ylabel(selected_y_var)
plt.title("Scatterplot of Palmer's Penguins")
st.pyplot(fig)

st.title('Pandas Profiling of Penguin Dataset')
penguin_profile = ProfileReport(penguins_df, explorative=True)
st_profile_report(penguin_profile)
