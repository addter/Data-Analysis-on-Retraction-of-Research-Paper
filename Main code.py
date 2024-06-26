import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
data = pd.read_csv('C:/Users/nayan/OneDrive/Desktop/Data Visualization/clean.csv')

# Initialize TF-IDF vectorizer for n-gram range from 1 to 20
tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 20))

# Apply TF-IDF vectorization to text columns
subject_features = tfidf_vectorizer.fit_transform(data['Subject']).toarray()
reason_features = tfidf_vectorizer.fit_transform(data['Reason']).toarray()

# Initialize OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)

# One-hot encode the categorical columns 'Paywalled' and 'ArticleType'
paywalled_features = ohe.fit_transform(data[['Paywalled']])
article_type_features = ohe.fit_transform(data[['ArticleType']])
journal_features = ohe.fit_transform(data[['Journal']])
publisher_features = ohe.fit_transform(data[['Publisher']])
country_features = ohe.fit_transform(data[['Country']])
author_features = ohe.fit_transform(data[['Author']])

# Initialize StandardScaler
scaler = StandardScaler()

# Scale the numerical columns 'CitationCount' and 'DaysBetween'
numerical_features = data[['CitationCount']].values
scaled_numerical_features = scaler.fit_transform(numerical_features)

# Combine all features into one feature matrix
combined_features = np.hstack([
    scaled_numerical_features,
    paywalled_features,
    article_type_features,
    subject_features,
    reason_features,
    journal_features,
    publisher_features,
    country_features,
    author_features
])

# Perform feature selection using VarianceThreshold
variance_selector = VarianceThreshold(threshold=0.01)
selected_combined_features = variance_selector.fit_transform(combined_features)

# Apply KMeans clustering to combined features
n_clusters = 10  # Adjust the number of clusters as necessary
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_features = kmeans.fit_transform(selected_combined_features)

# Initialize PCA with 2 components
pca = PCA(n_components=2)
pca_features = pca.fit_transform(cluster_features)
explained_variance = pca.explained_variance_ratio_

# Print explained variance by each principal component
print(f"Variance explained by PC1: {explained_variance[0] * 100:.2f}%")
print(f"Variance explained by PC2: {explained_variance[1] * 100:.2f}%")
print(f"Total variance explained by PC1 and PC2: {np.sum(explained_variance) * 100:.2f}%")

# Scatter plot of PCA results
plt.figure(figsize=(8, 6))
plt.scatter(pca_features[:, 0], pca_features[:, 1], alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Scatter Plot of PCA Results')
plt.grid(True)  # Optional: Adds a grid for better visualization of scale and distribution
plt.show()

kmeans_pca = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_pca.fit(pca_features)
pca_cluster_labels = kmeans_pca.labels_

# Scatter plot of PCA results with new KMeans clustering overlay
plt.figure(figsize=(8, 6))
scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], c=pca_cluster_labels, cmap='viridis', alpha=0.5)
plt.colorbar(scatter, label='Cluster Label')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Results with New KMeans Clustering Overlay')
plt.show()

# Assume 'DaysBetween' is the target variable (time to retraction)
X_train, X_test, y_train, y_test = train_test_split(pca_features, data['DaysBetween'], test_size=0.4, random_state=42)

# Initialize the Random Forest regressor
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
random_forest_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_rf = random_forest_model.predict(X_test)

# Calculate MSE and R² score for the Random Forest model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Print the results
print(f"Random Forest Mean Squared Error: {mse_rf:.2f}")
print(f"Random Forest R² Score: {r2_rf:.2f}")

# Optional: Display feature importance
feature_importances = random_forest_model.feature_importances_
print("Feature Importances:", feature_importances)

# Initialize and train a LinearRegression model
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = linear_regression_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output the results
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Display coefficients
print("Coefficients:", linear_regression_model.coef_)


