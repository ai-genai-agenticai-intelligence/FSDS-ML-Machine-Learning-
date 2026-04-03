# PCA-----------------

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import pandas as pd

#Example data(replace with your actual data)
data = load_iris()
X = data.data
y = data.target

# Apply PCA
pca =PCA(n_components=2)
# Fit and transfer the data
x_pca = pca.fit_transform(X)



#Visualize the reduction dimensionality data
# Visualize the reduced dimensionality data
sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=y, palette='viridis', s=50)

plt.title('PCA: Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Target')
plt.show()