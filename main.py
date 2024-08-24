import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# Title and description
st.title('AI-Driven Media Investment Plan')
st.sidebar.title("Navigation")

# Sidebar for Navigation
page = st.sidebar.selectbox("Choose a section", ["Upload Data", "Data Exploration", "Model Training", "Results"])

# Upload Data Section
if page == "Upload Data":
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

# Data Exploration Section
if page == "Data Exploration" and 'data' in locals():
    st.header("Explore your data")

    # Show DataFrame Overview
    st.subheader("Data Overview")
    st.write(data.describe())

    # Show the distribution of categorical features
    st.subheader("Categorical Feature Distribution")
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        st.write(f"Distribution of {col}:")
        st.bar_chart(data[col].value_counts())

    # Show Correlation Matrix
    st.subheader("Correlation Matrix")
    corr_matrix = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Example Filter: Date Range
    if 'Timestamp' in data.columns:
        st.subheader("Filter by Date Range")
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        start_date, end_date = st.date_input("Select Date Range:", [data['Timestamp'].min(), data['Timestamp'].max()])
        filtered_data = data[(data['Timestamp'] >= pd.to_datetime(start_date)) & (data['Timestamp'] <= pd.to_datetime(end_date))]
        st.write(filtered_data)

    # Interactive Plot with Plotly
    st.subheader("Interactive Plot: Revenue by Channel")
    if 'Revenue' in data.columns:
        fig = px.bar(data, x='ChannelSource', y='Revenue', title="Revenue by Channel Source")
        st.plotly_chart(fig)

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
