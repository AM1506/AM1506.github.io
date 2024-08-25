import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# Custom CSS for background color and font color
st.markdown(
    """
    <style>
    /* Background Color */
    body {
        background-color: #f0f2f6; /* Light gray background */
    }

    /* Font Color */
    h1, h2, h3, h4, h5, h6, p {
        color: #333333; /* Dark gray text */
    }

    /* Sidebar Background Color and Font Color */
    .sidebar .sidebar-content {
        background-color: #ffffff; /* White background for sidebar */
        color: #333333; /* Dark gray text for sidebar */
    }

    /* Input Text Color */
    input, textarea, select {
        color: #333333; /* Dark gray text for inputs */
    }

    /* Button Style */
    button {
        background-color: #4CAF50; /* Green button background */
        color: white; /* White button text */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.title('AI-Driven Media Investment Plan')
st.sidebar.title("Navigation")

# Sidebar for Navigation
page = st.sidebar.selectbox("Choose a section", ["Input Data & Budget", "Approach & Methodology", "Model Training", "Results"])

# Input Data & Budget Section
if page == "Input Data & Budget":
    st.header("Input Section")

    # New Budget as Input
    budget = st.number_input("Enter the New Budget Amount (in $)", min_value=0, step=1000)
    st.write(f"New Budget Amount: ${budget}")

    # Select and read one dataset
    st.header("Upload your dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Here's a preview of your dataset:")
        st.write(data.head())

        # Data Summary
        st.write("Dataset Summary:")
        st.write(f"**Number of Rows:** {data.shape[0]}")
        st.write(f"**Number of Columns:** {data.shape[1]}")
        st.write("**Column Data Types:**")
        st.write(data.dtypes)

# Approach and Methodology Section
if page == "Approach & Methodology":
    st.header("Approach and Methodology")

    st.subheader("Data Processing")
    st.write("Describe any data cleaning and preprocessing steps taken with the input data.")
    st.write("""
        - Handling missing values
        - Converting categorical variables to numeric
        - Scaling or normalizing features
        - Splitting data into training and testing sets
    """)

    st.subheader("Algorithm")
    st.write("Explain the algorithm used for budget allocation, including any mathematical formulas, logic, or ML Model.")
    st.write("""
        - Linear Regression for predicting potential revenue based on previous data
        - Random Forest and Gradient Boosting for more robust predictions
        - Mathematical formulas or logic used in calculating budget allocation
    """)

    st.subheader("Assumptions")
    st.write("List any assumptions made during the process.")
    st.write("""
        - Assuming that the provided dataset is clean and well-formatted
        - Assuming that the budget will be allocated based on historical performance
        - Assuming that the user inputs a reasonable budget amount that aligns with the dataset
    """)

# Model Training Section
if page == "Model Training" and 'data' in locals():
    st.header("Model Training")
    
    # Select features
    st.subheader("Select Features")
    features = st.multiselect("Select features for model training:", data.columns.tolist(), default=data.columns.tolist())

    target_column = st.selectbox("Select the target column", data.columns)

    if features and target_column:
        X = data[features]
        y = data[target_column]

        # Convert categorical variables to numeric (one-hot encoding)
        X = pd.get_dummies(X, drop_first=True)

        # Train/test split
        test_size = st.slider("Test set size (as a percentage)", 10, 50, 20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

        # Model Tuning Options
        st.subheader("Model Parameters")
        normalize = st.checkbox("Normalize the data", value=True)
        model_type = st.selectbox("Choose Model", ["Linear Regression", "Random Forest", "Gradient Boosting"])

        if model_type == "Linear Regression":
            model = LinearRegression(normalize=normalize)
        elif model_type == "Random Forest":
            n_estimators = st.slider("Number of trees:", 10, 500, 100)
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        elif model_type == "Gradient Boosting":
            learning_rate = st.slider("Learning Rate:", 0.01, 0.5, 0.1)
            model = GradientBoostingRegressor(learning_rate=learning_rate, random_state=42)

        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Display model performance
        st.write(f"### Mean Squared Error: {mse}")
        st.write(f"### R2 Score: {r2}")

# Results Section
if page == "Results" and 'data' in locals():
    st.header("Model Results")

    if 'y_pred' in locals():
        st.write(f"### Mean Squared Error: {mse}")
        st.write(f"### R2 Score: {r2}")

        st.write("### Prediction vs Actual")
        comparison = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        st.line_chart(comparison)

        # Feature Importances (if applicable)
        if model_type in ["Random Forest", "Gradient Boosting"]:
            st.subheader("Feature Importances")
            feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            st.bar_chart(feature_importances)

    else:
        st.write("Train the model first to see the results.")
