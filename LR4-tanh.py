import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

st.set_page_config(page_title="Tanh Activation Function", layout="centered")

st.title("Hyperbolic Tangent (tanh) Activation Function Visualisation")
st.write("### f(x) = tanh(x)")

x_min, x_max = st.slider("Select x-range", -10.0, 10.0, (-6.0, 6.0))

x = torch.linspace(x_min, x_max, 400)
tanh = nn.Tanh()
y = tanh(x)

fig, ax = plt.subplots()
ax.plot(x.numpy(), y.numpy())
ax.axhline(1, linestyle="--")
ax.axhline(-1, linestyle="--")
ax.axvline(0)
ax.set_xlabel("Input x")
ax.set_ylabel("Output f(x)")
ax.set_title("Tanh Activation Function")

st.pyplot(fig)