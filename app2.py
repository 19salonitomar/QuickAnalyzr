# Imports
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# App configuration
st.set_page_config(layout="wide")
st.title("📊 Advanced Data Analysis Tool")
st.subheader("Built with Python, Streamlit & Machine Learning")

# Upload section
upload = st.file_uploader("📂 Upload your Dataset (CSV Format)")
if upload is not None:
    data = pd.read_csv(upload)
    st.success("✅ Dataset Loaded Successfully!")

    # Define numeric data globally
    numeric_data = data.select_dtypes(include=['float64', 'int64'])

    # Sidebar options
    st.sidebar.header("⚙️ Choose Analysis Options")

    # Preview
    if st.sidebar.checkbox("Preview Dataset"):
        st.subheader("🔍 Dataset Preview")
        st.write(data.head())

    # Data Types
    if st.sidebar.checkbox("Show Data Types"):
        st.subheader("🧬 Data Types")
        st.write(data.dtypes)

    # Dataset Shape
    if st.sidebar.checkbox("Show Shape"):
        st.info(f"📏 Rows: {data.shape[0]}, Columns: {data.shape[1]}")

    # Null Value Heatmap
    if data.isnull().values.any():
        if st.sidebar.checkbox("Show Null Value Heatmap"):
            st.subheader("🟡 Missing Values Heatmap")
            fig, ax = plt.subplots()
            sns.heatmap(data.isnull(), cbar=False, cmap="viridis", ax=ax)
            st.pyplot(fig)

    # Handle Missing Values
    if st.sidebar.checkbox("Impute Missing Values"):
        numeric_cols = numeric_data.columns.tolist()
        if numeric_cols:
            imputer = SimpleImputer(strategy='mean')
            data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
            st.success("✅ Missing values imputed using mean strategy.")
            st.write(data)
        else:
            st.warning("⚠️ No numeric columns found to impute.")

    # Drop Duplicates
    if data.duplicated().any():
        if st.sidebar.checkbox("Remove Duplicates"):
            data = data.drop_duplicates()
            st.success("✅ Duplicates Removed")

    # Summary
    if st.sidebar.checkbox("Show Summary Statistics"):
        st.subheader("📈 Descriptive Statistics")
        st.write(data.describe(include='all'))

    # Correlation Heatmap
    if st.sidebar.checkbox("Correlation Matrix"):
        st.subheader("🔗 Correlation Matrix")
        if not numeric_data.empty and numeric_data.shape[1] > 1:
            corr = numeric_data.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("⚠️ Not enough numeric columns for correlation matrix.")

    # Boxplots
    if st.sidebar.checkbox("Show Boxplots"):
        st.subheader("📦 Boxplots")
        if not numeric_data.empty:
            for col in numeric_data.columns:
                fig, ax = plt.subplots()
                sns.boxplot(x=data[col], ax=ax)
                st.pyplot(fig)
        else:
            st.warning("⚠️ No numeric columns available for boxplot.")

    # Histograms
    if st.sidebar.checkbox("Show Histograms"):
        st.subheader("📊 Histograms")
        if not numeric_data.empty:
            for col in numeric_data.columns:
                fig, ax = plt.subplots()
                sns.histplot(data[col], kde=True, ax=ax)
                st.pyplot(fig)
        else:
            st.warning("⚠️ No numeric columns available for histogram.")

    # Geo Map Plot (if latitude & longitude)
    if st.sidebar.checkbox("Show Map (if Geo Data present)"):
        geo_cols = ['latitude', 'lat', 'Longitude', 'longitude', 'long']
        if 'latitude' in data.columns and 'longitude' in data.columns:
            st.subheader("🗺️ Geographical Plot")
            st.map(data[['latitude', 'longitude']])
        elif set(['lat', 'long']).issubset(data.columns):
            st.subheader("🗺️ Geographical Plot (lat/long renamed)")
            st.map(data.rename(columns={'lat': 'latitude', 'long': 'longitude'})[['latitude', 'longitude']])
        else:
            st.warning("⚠️ No geo-coordinates found in dataset.")

    # Filter by Column
    if st.sidebar.checkbox("Filter By Column"):
        column = st.selectbox("Select Column", data.columns)
        unique_vals = data[column].unique()
        selected = st.selectbox("Select Value", unique_vals)
        st.write(data[data[column] == selected])

    # Normalize Data
    if st.sidebar.checkbox("Standardize Data"):
        st.subheader("⚖️ Standardized Data")
        if not numeric_data.empty:
            scaler = StandardScaler()
            data_scaled = data.copy()
            data_scaled[numeric_data.columns] = scaler.fit_transform(numeric_data)
            st.write(data_scaled)
        else:
            st.warning("⚠️ No numeric columns to standardize.")

    # ML Classification
    if st.sidebar.checkbox("Run ML Classification"):
        st.subheader("🤖 Machine Learning: Random Forest Classifier")
        target_column = st.selectbox("🎯 Select Target Column", data.columns)
        if st.button("🚀 Train Model"):
            X = data.drop(columns=[target_column])
            y = data[target_column]

            # Convert categoricals
            X = pd.get_dummies(X)
            if y.dtype == 'object':
                y = pd.factorize(y)[0]

            if X.shape[1] < 1:
                st.error("❌ Not enough features to train a model.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                st.success(f"✅ Model Trained! Accuracy: {acc:.2f}")
                st.text("📋 Classification Report:")
                st.text(classification_report(y_test, preds))

    # About Section
    if st.sidebar.button("About"):
        st.info("📌 Built using Streamlit, Pandas, Seaborn, Sklearn, and Matplotlib.\n\n👨‍💻 Created for interactive data exploration & ML.\n\n🧑‍🎓 BY: Saloni Tomar")
