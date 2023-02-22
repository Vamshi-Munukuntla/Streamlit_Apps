import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title('Illustrating the Central Limit Theorem with Streamlit')
st.subheader('Created by M. Vamshi')
st.write('This app simulates a 1000 coin flips using the chance of heads '
         'below, and then samples with replacement from that population and '
         'plots the histogram of the means of the samples, in order to '
         'illustrate the Central Limit Theorem')

perc_heads = st.number_input(label='Chance of Coins Landing on Heads',
                             min_value=0.0, max_value=1.0, value=0.5)

# Generating a binomial distribution we want to sample from with replacement
binom_dist = np.random.binomial(1, perc_heads, 1000)


# From Central Limit Theorem (CLT), if we sampled enough times from binom_dist,
# the mean of those samples would approximate the normal distribution.

list_of_means = []
for i in range(0, 1000):
    list_of_means.append(np.random.choice(binom_dist, 100,
                                          replace=True).mean())

fig, ax = plt.subplots()
ax = plt.hist(list_of_means)
st.pyplot(fig)

