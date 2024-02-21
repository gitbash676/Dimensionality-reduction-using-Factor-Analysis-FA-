# Dimensionality-reduction-using-Factor-Analysis-FA-
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

# Load iris dataset
data = load_iris()
x = data.data
feature_names = data.feature_names

# Standardize the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Apply Factor Analysis
n_components = 2
fa = FactorAnalysis(n_components=n_components, random_state=42)
fa.fit(x_scaled)

# Print factor loadings
for i in range(n_components):
    print(f"Factors{i+1}:")
    for j, feature in enumerate(feature_names):
        print(f"{feature}: {fa.components_[i, j]}")
    print()
