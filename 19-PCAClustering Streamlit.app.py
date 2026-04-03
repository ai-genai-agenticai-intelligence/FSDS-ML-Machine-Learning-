# PCA Clustering Streamlit ---------------

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Title
st.title(" PCA Visualization - Iris Dataset")

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Sidebar options
st.sidebar.header("Settings")
n_components = st.sidebar.slider("Select number of PCA components", 2, 4, 2)

# Apply PCA
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Show explained variance
st.subheader(" Explained Variance Ratio")
st.write(pca.explained_variance_ratio_)

# Create DataFrame
df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
df['Target'] = y

# Plot
st.subheader(" PCA Scatter Plot")

fig, ax = plt.subplots()
sns.scatterplot(
    x=df['PC1'],
    y=df['PC2'],
    hue=df['Target'],
    palette='viridis',
    s=60,
    ax=ax
)

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_title("PCA: Iris Dataset")

st.pyplot(fig)

# Show dataset
if st.checkbox("Show Raw Data"):
    st.subheader(" Dataset")
    st.write(pd.DataFrame(X, columns=data.feature_names))

