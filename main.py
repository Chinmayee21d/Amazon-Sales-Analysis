import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import numpy as np

# Set the page config with Amazon-themed colors
st.set_page_config(
    page_title="Amazon Sales",
    page_icon="🛒",
    layout="wide",
)
# Load and preprocess the data
@st.cache_data
def load_data():
    data = pd.read_csv('C:/Users/Chinmayee/PycharmProjects/Q/Amazon Sales/Amazonsales.csv')

    # Clean the Sales and Profit columns (remove $ and commas, convert to float)
    data['Sales'] = data['Sales'].replace({'\$': '', ',': ''}, regex=True).astype(float)
    data['Profit'] = data['Profit'].replace({'\$': '', ',': ''}, regex=True).astype(float)

    # Convert Order Date and Ship Date to datetime
    data['Order Date'] = pd.to_datetime(data['Order Date'], format='%d/%m/%Y', errors='coerce')
    data['Ship Date'] = pd.to_datetime(data['Ship Date'], format='%d/%m/%Y', errors='coerce')

    # Create Year and Month columns
    data['Year'] = data['Order Date'].dt.year
    data['Month'] = data['Order Date'].dt.month

    return data


data = load_data()

# Modify year values for specific years
#year Mapping
data['Year'] = data['Year'].replace({2012: 2020, 2013: 2021, 2014: 2022, 2015: 2023})

# Application title and logo
st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=200)  # Amazon logo URL
st.title("Amazon Sales Analysis Dashboard")

# Sidebar section for navigation
menu = st.sidebar.radio("Navigation",
                        ["Home", "Product-Based Analysis", "Sales Analysis", "Classification", "K-Means Clustering"])

# Home Page: Current Year Sales & Profit Visualization
if menu == "Home":
    st.title("Sales and Profit Analysis")

    current_year = data['Year'].max()
    current_year_data = data[data['Year'] == current_year]

    # Displaying current year sales
    st.subheader(f"Sales for {current_year}")
    yearly_sales = current_year_data.groupby('Month')['Sales'].sum().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(data=yearly_sales, x='Month', y='Sales', palette="Set2", ax=ax)
    st.pyplot(fig)

    # Total profit
    total_profit = current_year_data['Profit'].sum()
    st.subheader(f"Total Profit for {current_year}: ${total_profit:,.2f}")

    # Profit visualization
    st.subheader(f"Profit Visualization for {current_year}")
    monthly_profit = current_year_data.groupby('Month')['Profit'].sum().reset_index()
    fig, ax = plt.subplots()
    sns.lineplot(data=monthly_profit, x='Month', y='Profit', marker='o', color='green', ax=ax)
    st.pyplot(fig)

# Product-Based Analysis
if menu == "Product-Based Analysis":
    st.title("Product-Based Analysis")
    analysis_option = st.selectbox("Select Product Analysis Option",
                                   ["Product with Highest Sales", "Category-Based Sales", "Top Products by Sales"])

    if analysis_option == "Product with Highest Sales":
        st.subheader("Product with Highest Sales")
        top_product = data.groupby('Product Name')['Sales'].sum().idxmax()
        st.write(f"The product with the highest sales is: **{top_product}**")

        # Bar chart of top products by sales
        st.subheader("Top 10 Products by Sales")
        top_products = data.groupby('Product Name')['Sales'].sum().nlargest(10).reset_index()
        fig, ax = plt.subplots()
        sns.barplot(data=top_products, x='Sales', y='Product Name', palette="coolwarm", ax=ax)
        st.pyplot(fig)

    elif analysis_option == "Category-Based Sales":
        st.subheader("Sales by Category")
        category_sales = data.groupby('Category')['Sales'].sum().reset_index()
        fig, ax = plt.subplots()
        sns.barplot(data=category_sales, x='Sales', y='Category', palette="viridis", ax=ax)
        st.pyplot(fig)

# Sales Analysis
if menu == "Sales Analysis":
    st.title("Sales Analysis")
    analysis_option = st.selectbox("Select Yearly Analysis Option", ["Yearly Sales", "Yearly Profit", "Total Sales"])

    if analysis_option == "Yearly Sales":
        st.subheader("Yearly Sales Analysis")
        yearly_sales = data.groupby('Year')['Sales'].sum().reset_index()
        fig, ax = plt.subplots()
        sns.lineplot(data=yearly_sales, x='Year', y='Sales', marker='o', color='purple', ax=ax)
        st.pyplot(fig)

    elif analysis_option == "Yearly Profit":
        st.subheader("Yearly Profit Analysis")
        yearly_profit = data.groupby('Year')['Profit'].sum().reset_index()
        fig, ax = plt.subplots()
        sns.lineplot(data=yearly_profit, x='Year', y='Profit', marker='o', color='blue', ax=ax)
        st.pyplot(fig)

    elif analysis_option == "Total Sales":
        total_sales = data['Sales'].sum()
        st.write(f"Total Sales: ${total_sales:,.2f}")

# Classification Section
if menu == "Classification":
    st.title("Classification Models")

    # Dropdown to select classification model
    classifier = st.selectbox("Select Classification Model", ["Naive Bayes", "Decision Tree"])

    # Prepare data for classification
    features = data[['Sales', 'Profit', 'Quantity', 'Discount', 'Shipping Cost', 'Order Priority', 'Category']]
    target = data['Segment']

    # Encoding categorical variables
    label_encoder = LabelEncoder()
    features['Order Priority'] = label_encoder.fit_transform(features['Order Priority'])
    features['Category'] = label_encoder.fit_transform(features['Category'])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Naive Bayes Classifier
    if classifier == "Naive Bayes":
        st.subheader("Naive Bayes Classifier Results")
        model = GaussianNB()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        st.write(f"Accuracy: {accuracy:.2f}")
        st.text(classification_report(y_test, predictions))

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, predictions)
        fig, ax = plt.subplots()
        sns.barplot(data=pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_), ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        # Bar graph for accuracy
        st.subheader("Model Accuracy")
        fig, ax = plt.subplots()
        sns.barplot(x=['Naive Bayes'], y=[accuracy], ax=ax, palette='Blues')
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        st.pyplot(fig)

    # Decision Tree Classifier
    elif classifier == "Decision Tree":
        st.subheader("Decision Tree Classifier Results")
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        st.write(f"Accuracy: {accuracy:.2f}")
        st.text(classification_report(y_test, predictions))

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, predictions)
        fig, ax = plt.subplots()
        sns.barplot(data=pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_), ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        # Bar graph for accuracy
        st.subheader("Model Accuracy")
        fig, ax = plt.subplots()
        sns.barplot(x=['Decision Tree'], y=[accuracy], ax=ax, palette='Oranges')
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        st.pyplot(fig)

# K-Means Clustering Section
if menu == "K-Means Clustering":
    st.title("K-Means Clustering Analysis")

    # Prepare data for K-means
    features_kmeans = data[['Sales', 'Profit', 'Quantity', 'Discount', 'Shipping Cost']]

    # Selecting the number of clusters
    n_clusters = st.slider("Select number of clusters", 2, 10, 3)

    # Applying K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    features_kmeans['Cluster'] = kmeans.fit_predict(features_kmeans)

    # Visualization
    fig, ax = plt.subplots()
    sns.scatterplot(data=features_kmeans, x='Sales', y='Profit', hue='Cluster', palette='Set2', ax=ax)
    ax.set_title(f'K-Means Clustering with {n_clusters} Clusters')
    st.pyplot(fig)

    # Display cluster centers
    st.subheader("Cluster Centers")
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_,
                                   columns=['Sales', 'Profit', 'Quantity', 'Discount', 'Shipping Cost'])
    st.write(cluster_centers)
