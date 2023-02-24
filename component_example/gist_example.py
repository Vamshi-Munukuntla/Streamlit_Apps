import streamlit as st
from streamlit_embedcode import github_gist

st.title('Github Gist Example')
st.write("Code from Palmer's Penguin Streamlit app.")
github_gist('https://gist.github.com/Vamshi-Munukuntla/75a4001e8132f929e12ce49fb59778ba')

