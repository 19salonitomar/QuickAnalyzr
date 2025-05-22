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
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import plotly.figure_factory as ff

# App Configuration
st.set_page_config(layout="wide")
st.title("📊 Advanced Data Analysis & Machine Learning App")
st.subheader("Built with Python, Streamlit, and Scikit-Learn")

# Upload Section
upload = st.file_uploader("📂 Upload your Dataset (CSV Format)")
if upload is not None:
    data = pd.read_csv(upload)
    st.success("✅ Dataset Loaded Successfully!")

    # Fix Arrow Compatibility
    for col in data.columns:
        if pd.api.types.is_string_dtype(data[col]) or pd.api.types.is_object_dtype(data[col]):
            data[col] = data[col].astype(str)
        elif pd.api.types.is_integer_dtype(data[col]):
            data[col] = data[col].astype('int64')
        elif pd.api.types.is_bool_dtype(data[col]):
            data[col] = data[col].astype(bool)

    numeric_data = data.select_dtypes(include=['float64', 'int64'])

    # Sidebar Controls
    st.sidebar.header("⚙️ Choose Analysis Options")

    if st.sidebar.checkbox("Preview Dataset"):
        st.subheader("🔍 Dataset Preview")
        st.write(data.head())

    if st.sidebar.checkbox("Show Data Types"):
        st.subheader("🧬 Data Types")
        st.write(data.dtypes)

    if st.sidebar.checkbox("Show Shape"):
        st.info(f"📏 Rows: {data.shape[0]}, Columns: {data.shape[1]}")

    if data.isnull().values.any() and st.sidebar.checkbox("Show Null Value Heatmap"):
        st.subheader("🟡 Missing Values Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(data.isnull(), cbar=False, cmap="viridis", ax=ax)
        st.pyplot(fig)

    if st.sidebar.checkbox("Impute Missing Values"):
        # Separate numeric and non-numeric columns
        num_cols = data.select_dtypes(include=['float64', 'int64']).columns
        cat_cols = data.select_dtypes(include=['object', 'string']).columns

        if not num_cols.empty:
            data[num_cols] = SimpleImputer(strategy='mean').fit_transform(data[num_cols])
        if not cat_cols.empty:
            data[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(data[cat_cols])

        st.success("✅ Missing values imputed (numeric: mean, categorical: mode).")

    if data.duplicated().any() and st.sidebar.checkbox("Remove Duplicates"):
        data = data.drop_duplicates()
        st.success("✅ Duplicates Removed")

    if st.sidebar.checkbox("Show Summary Statistics"):
        st.subheader("📈 Descriptive Statistics")
        st.write(data.describe(include='all'))

    if st.sidebar.checkbox("Correlation Matrix"):
        st.subheader("🔗 Correlation Matrix")
        if not numeric_data.empty and numeric_data.shape[1] > 1:
            corr = data[numeric_data.columns].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("⚠️ Not enough numeric columns for correlation matrix.")

    if st.sidebar.checkbox("Show Boxplots"):
        st.subheader("📦 Boxplots")
        for col in numeric_data.columns:
            fig, ax = plt.subplots()
            sns.boxplot(x=data[col], ax=ax)
            st.pyplot(fig)

    if st.sidebar.checkbox("Show Histograms"):
        st.subheader("📊 Histograms")
        for col in numeric_data.columns:
            fig, ax = plt.subplots()
            sns.histplot(data[col], kde=True, ax=ax)
            st.pyplot(fig)

    if st.sidebar.checkbox("Show Map (if Geo Data present)"):
        if 'latitude' in data.columns and 'longitude' in data.columns:
            st.subheader("🗺️ Geographical Plot")
            st.map(data[['latitude', 'longitude']])
        elif set(['lat', 'long']).issubset(data.columns):
            st.subheader("🗺️ Geographical Plot (lat/long renamed)")
            st.map(data.rename(columns={'lat': 'latitude', 'long': 'longitude'})[['latitude', 'longitude']])
        else:
            st.warning("⚠️ No geo-coordinates found in dataset.")

    if st.sidebar.checkbox("Filter By Column"):
        column = st.selectbox("Select Column", data.columns)
        selected = st.selectbox("Select Value", data[column].unique())
        st.write(data[data[column] == selected])

    if st.sidebar.checkbox("Standardize Data"):
        st.subheader("⚖️ Standardized Data")
        if not numeric_data.empty:
            scaler = StandardScaler()
            data_scaled = data.copy()
            data_scaled[numeric_data.columns] = scaler.fit_transform(numeric_data)
            st.write(data_scaled)
        else:
            st.warning("⚠️ No numeric columns to standardize.")

    if st.sidebar.checkbox("Run ML Classification"):
        st.subheader("🤖 Machine Learning: Classifier Comparison")
        target_column = st.selectbox("🎯 Select Target Column", data.columns)

        model_choice = st.selectbox("Choose ML Model", [
            "Random Forest", "Support Vector Machine", 
            "Logistic Regression", "K-Nearest Neighbors", "Naive Bayes"
        ])

        if st.button("🚀 Train Model"):
            X = data.drop(columns=[target_column])
            y = data[target_column]

            # Handle NaNs again just in case
            X = pd.get_dummies(X)
            X = SimpleImputer(strategy='mean').fit_transform(X)

            if y.dtype == 'object' or isinstance(y.iloc[0], str):
                y = pd.factorize(y)[0]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            if model_choice == "Random Forest":
                model = RandomForestClassifier()
            elif model_choice == "Support Vector Machine":
                model = SVC()
            elif model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif model_choice == "K-Nearest Neighbors":
                model = KNeighborsClassifier()
            elif model_choice == "Naive Bayes":
                model = GaussianNB()

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            st.success(f"✅ {model_choice} Accuracy: {acc:.2f}")
            st.text("📋 Classification Report:")
            st.text(classification_report(y_test, preds))

            cm = confusion_matrix(y_test, preds)
            st.subheader("🧩 Confusion Matrix")
            fig = ff.create_annotated_heatmap(
                z=cm,
                x=[str(i) for i in range(cm.shape[1])],
                y=[str(i) for i in range(cm.shape[0])],
                colorscale='Viridis'
            )
            st.plotly_chart(fig)

    if st.sidebar.button("About"):
        st.info("""
📌 Built using Streamlit, Pandas, Seaborn, Sklearn, and Matplotlib.
👨‍💻 Created for interactive data exploration & ML.
🧑‍🎓 BY: Saloni Tomar
        """)
