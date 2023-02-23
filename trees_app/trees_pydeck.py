import streamlit as st
import pandas as pd
import pydeck as pdk

st.title('SF Trees')
st.write('This app analyses trees in San Francisco using'
         ' a dataset kindly provided by SF DPW')

trees_df = pd.read_csv(r'.\trees_app\trees.csv')
trees_df.dropna(how='any', inplace=True)

pitch = st.slider(label='pitch', min_value=0, max_value=180)
radius = st.slider(label='radius', min_value=0, max_value=180)

sf_initial_view = pdk.ViewState(
    latitude=37.77,
    longitude=-122.4,
    zoom=11,
    pitch=pitch
)
st.write(pitch)

hx_layer = pdk.Layer(
    'HexagonLayer',
    data=trees_df,
    get_position=['longitude', 'latitude'],
    radius=radius,
    extruded=True)

sp_layer = pdk.Layer(
    'ScatterplotLayer',
    data=trees_df,
    get_position=['longitude', 'latitude'],
    get_radius=30)

st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=sf_initial_view,
    layers=[hx_layer]
))
