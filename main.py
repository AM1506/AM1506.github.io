pip install scikit-learn

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Title and description
st.title('AI-Driven Media Investment Plan')
st.write('Upload your dataset and analyze the performance of various channels in the customer journey.')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)

    # Display the first few rows of the dataset
    st.write("Here's a preview of your dataset:")
    st.write(data.head())

    # Basic preprocessing - this can be expanded based on your specific requirements
    if 'Timestamp' in data.columns:
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        data['Hour'] = data['Timestamp'].dt.hour
        data['DayOfWeek'] = data['Timestamp'].dt.dayofweek

    # Select the target column (assuming 'Revenue' is the target)
    target_column = st.selectbox("Select the target column", data.columns)
    
    if target_column:
        X = data.drop(columns=[target_column, 'CustomerID', 'Timestamp'], errors='ignore')
        y = data[target_column]

        # Convert categorical variables to numeric (one-hot encoding)
        X = pd.get_dummies(X, drop_first=True)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Display results
        st.write(f"### Results:")
        st.write(f"**Mean Squared Error:** {mse}")
        st.write(f"**R2 Score:** {r2}")

        # Optionally, plot the predictions vs actuals
        st.write("### Prediction vs Actual")
        comparison = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        st.line_chart(comparison)

