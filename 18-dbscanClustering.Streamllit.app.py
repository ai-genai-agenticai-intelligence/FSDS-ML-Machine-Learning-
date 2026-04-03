#DBSCAN Streamlit.py-------------------------

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

st.title("DBSCAN Clustering Demo")

# Sidebar inputs
st.sidebar.header("Parameters")
eps = st.sidebar.slider("Epsilon (eps)", 0.1, 1.0, 0.3)
min_samples = st.sidebar.slider("Min Samples", 1, 20, 10)

# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(
    n_samples=750,
    centers=centers,
    cluster_std=0.4,
    random_state=0
)

X = StandardScaler().fit_transform(X)

# Apply DBSCAN
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
labels = db.labels_

# Core samples mask
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

# Metrics
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

st.subheader("Clustering Results")
st.write(f"Estimated number of clusters: {n_clusters_}")
st.write(f"Estimated number of noise points: {n_noise_}")
st.write(f"Homogeneity: {metrics.homogeneity_score(labels_true, labels):.3f}")
st.write(f"Completeness: {metrics.completeness_score(labels_true, labels):.3f}")
st.write(f"V-measure: {metrics.v_measure_score(labels_true, labels):.3f}")
st.write(f"Adjusted Rand Index: {metrics.adjusted_rand_score(labels_true, labels):.3f}")
st.write(f"Adjusted Mutual Info: {metrics.adjusted_mutual_info_score(labels_true, labels):.3f}")

# Silhouette score (only if clusters > 1)
if n_clusters_ > 1:
    silhouette = metrics.silhouette_score(X, labels)
    st.write(f"Silhouette Coefficient: {silhouette:.3f}")
else:
    st.write("Silhouette Coefficient: Not applicable")

# Plot
st.subheader("Cluster Visualization")

fig, ax = plt.subplots()

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]  # Noise = Black

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    ax.plot(xy[:, 0], xy[:, 1], 'o',
            markerfacecolor=tuple(col),
            markeredgecolor='k',
            markersize=10)

    xy = X[class_member_mask & ~core_samples_mask]
    ax.plot(xy[:, 0], xy[:, 1], 'o',
            markerfacecolor=tuple(col),
            markeredgecolor='k',
            markersize=5)

ax.set_title(f'Estimated clusters: {n_clusters_}')
st.pyplot(fig)