import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the customer dataset
data = pd.read_csv("customers.csv")

# Select features for clustering
X = data[["annual_income_k", "spending_score"]]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train K-Means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Get cluster centers
centers = kmeans.cluster_centers_

# Calculate the average income and spending for each cluster
cluster_summary = data.copy()
cluster_summary["cluster"] = kmeans.labels_

summary = cluster_summary.groupby("cluster")[
    ["annual_income_k", "spending_score"]
].mean()

print("Cluster Summary:")
print(summary)

# Assign marketing personas based on cluster characteristics
personas = {}

for cluster in summary.index:
    income = summary.loc[cluster, "annual_income_k"]
    spending = summary.loc[cluster, "spending_score"]

    if income >= summary["annual_income_k"].median() and spending >= summary["spending_score"].median():
        personas[cluster] = "High Value Customers"
    elif income >= summary["annual_income_k"].median() and spending < summary["spending_score"].median():
        personas[cluster] = "High Income Low Spenders"
    else:
        personas[cluster] = "Budget Conscious Customers"

print("\nMarketing Personas:")
print(personas)

# Save the trained model and scaler
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(personas, "personas.pkl")

print("\nK-Means model saved as kmeans_model.pkl")
print("Scaler saved as scaler.pkl")
print("Personas saved as personas.pkl")