# streamlit_app.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Title of the app
st.title('CSV Uploader and MeanShift Clustering')

# Instructions for the user
st.write("Please upload a CSV file to perform clustering")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Check if a file has been uploaded
if uploaded_file is not None:
    # Load the CSV into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Display the first few rows of the DataFrame
    st.write("First few rows of the uploaded file:")
    st.write(df.head())

    # Select columns for clustering (let the user choose)
    selected_columns = st.multiselect('Select columns for clustering', df.columns.tolist())

    if len(selected_columns) > 0:
        # Extract the physical attributes chosen by the user
        physical_attributes = df[selected_columns]

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(physical_attributes)

        # Perform PCA to reduce dimensionality to 2D for visualization
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(X_scaled)

        # Plot the PCA components
        st.write("PCA Visualization of the selected columns")
        pca_fig = px.scatter(x=data_pca[:, 0], y=data_pca[:, 1], title="PCA Components", labels={"x": "PCA 1", "y": "PCA 2"})
        st.plotly_chart(pca_fig)

        # Perform Mean-Shift Clustering
        st.write("Performing Mean-Shift Clustering...")
        bandwidth = st.slider('Select Bandwidth for MeanShift', 0.1, 2.0, 1.0, step=0.1)
        mean_shift = MeanShift(bandwidth=bandwidth)
        labels = mean_shift.fit_predict(X_scaled)

        # Show the clustering results
        df['Cluster'] = labels
        st.write("Data with Cluster Labels:")
        st.write(df)

        # Plot the clusters
        st.write("Cluster Visualization")
        cluster_fig = px.scatter(x=data_pca[:, 0], y=data_pca[:, 1], color=labels, title="Clusters (MeanShift)")
        st.plotly_chart(cluster_fig)

        # Calculate and display clustering evaluation metrics
        silhouette_avg = silhouette_score(X_scaled, labels)
        calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
        davies_bouldin = davies_bouldin_score(X_scaled, labels)

        st.write(f"Silhouette Score: {silhouette_avg}")
        st.write(f"Calinski-Harabasz Index: {calinski_harabasz}")
        st.write(f"Davies-Bouldin Index: {davies_bouldin}")
