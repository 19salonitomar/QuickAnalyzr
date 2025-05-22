# 📊 QuickAnalyzr: Data Analysis and Visualization App

## 🚀 Overview

**QuickAnalyzr** is an interactive web-based tool built with **Streamlit** that allows users to effortlessly upload, explore, clean, visualize, and analyze datasets. Whether you're a beginner or an experienced data analyst, this tool makes **exploratory data analysis (EDA)** and basic **machine learning** accessible without writing any code.

---

## ✨ Features

- 📂 **Upload and Preview Datasets**
  - Upload your dataset in CSV format.
  - View the head or tail of the dataset.
  
- 🧼 **Handle Missing Data**
  - Automatically detects null values.
  - Visualizes missing data using heatmaps.
  - Impute missing values using the **mean strategy** for numerical columns.

- 🔁 **Remove Duplicates**
  - Identifies and removes duplicate rows.

- 📊 **Visualizations**
  - Histograms for distributions
  - Boxplots for outlier detection
  - Heatmaps for correlation
  - Interactive maps (if `latitude/longitude` columns exist)

- 📈 **Descriptive Statistics**
  - Summary statistics (mean, median, std, etc.)
  - Data type overview

- 🧪 **Data Normalization**
  - Standardize numerical features using **StandardScaler**

- 🤖 **Machine Learning Integration**
  - Run a **Random Forest Classifier**
  - Auto train/test split
  - View **accuracy score** and **classification report**

---


## 🧠 Tech Stack & Libraries

| Library        | Purpose                                                    |
|----------------|------------------------------------------------------------|
| `streamlit`    | Interactive UI for web-based apps                          |
| `pandas`       | DataFrame operations and CSV loading                       |
| `numpy`        | Backend numeric operations                                 |
| `matplotlib`   | Basic plotting support                                     |
| `seaborn`      | Advanced visualizations (heatmaps, boxplots, histograms)   |
| `plotly`       | Interactive map plotting (optional, for geo-coordinates)   |
| `scikit-learn` | Imputation, Standardization, ML models (Random Forest)     |

---

## 🧰 Installation

1. **Clone the repository:**

git clone https://github.com/yourusername/QuickAnalyzr.git
cd QuickAnalyzr


2. **Install the required packages:**

pip install -r requirements.txt

3. **(Optional) Create and activate a virtual environment:**

- Windows:

python -m venv venv
.\venv\Scripts\activate

- macOS/Linux:

python3 -m venv venv
source venv/bin/activate


## ▶️ Running the App
Once your environment is set up and dependencies are installed:
streamlit run APP1.py


##🧩 Project Structure

QuickAnalyzr/
│
├── app1.py               # Main Streamlit application
├── requirements.txt      # Required dependencies
└── README.md             # You're reading it!


## 📈 Use Cases
- Business Analysts exploring sales/customer data.

- Students learning data science and ML concepts.

- Teachers demonstrating data exploration workflows.

- Data scientists prototyping EDA pipelines.


## 💡 Future Enhancements
- Add more ML algorithms (e.g. Logistic Regression, SVM)

- Regression and clustering support

- Export cleaned/processed datasets

- Integrate pandas-profiling for auto-reports

- Deploy via Docker or Streamlit Cloud


## 🙏 Acknowledgements
Built using:

🐍 Python

📊 Streamlit, Pandas, Seaborn, Scikit-learn, Matplotlib

## 💻 Open-source community inspiration

📬 Contact
For feedback, issues, or contributions:
📧 salonitomar5813@gmail.com.com
🔗 GitHub

## Thank you for using QuickAnalyzr!
** Data analysis just got easier. 🚀**