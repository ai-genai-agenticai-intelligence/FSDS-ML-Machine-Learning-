import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

st.title("Hierarchical Clustering (Customer Segmentation)")

# Upload dataset
uploaded_file = st.file_uploader(r'D:\Ml-MACHINE LEARNING DATA\Mall_Customers.csv', type=["csv"])

if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", dataset.head())

    # Select features
    X = dataset.iloc[:, [3, 4]].values

    st.subheader("Dendrogram")
    fig1 = plt.figure()
    dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
    plt.title('Dendrogram')
    plt.xlabel('Customers')
    plt.ylabel('Distance')
    st.pyplot(fig1)

    # Select number of clusters
    n_clusters = st.slider("Select number of clusters", 2, 10, 5)

    # Apply Agglomerative Clustering
    hc = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    y_hc = hc.fit_predict(X)

    # Visualization
    st.subheader("Cluster Visualization")
    fig2 = plt.figure()

    colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown']

    for i in range(n_clusters):
        plt.scatter(X[y_hc == i, 0], X[y_hc == i, 1],
                    s=100, c=colors[i], label=f'Cluster {i+1}')

    plt.title('Clusters of Customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()

    st.pyplot(fig2)