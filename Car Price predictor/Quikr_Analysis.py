import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import streamlit as st
import seaborn as sns
import pickle

# Set the style for plots
mpl.style.use('ggplot')

# Read the car data
car = pd.read_csv('quikr_car.csv')

# Create a backup copy
backup = car.copy()

# Clean the data (same cleaning steps as in your code)

# Display the cleaned data
st.title('Cleaned Car Data')
st.write(car)

# Create a Streamlit app for the data analysis and prediction
st.sidebar.header('Car Price Predictor')

# Display some information about the data
st.sidebar.subheader('Data Information')
st.sidebar.write(f'Shape of the data: {car.shape}')
st.sidebar.write('Data Description:')
st.sidebar.write(car.describe(include='all'))

# Data analysis and visualization
st.header('Data Analysis and Visualization')

# Plot the relationship of Company with Price
st.subheader('Relationship of Company with Price')
fig, ax = plt.subplots(figsize=(15, 7))
sns.boxplot(x='company', y='Price', data=car)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
st.pyplot(fig)

# Plot the relationship of Year with Price
st.subheader('Relationship of Year with Price')
fig, ax = plt.subplots(figsize=(20, 10))
sns.swarmplot(x='year', y='Price', data=car)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
st.pyplot(fig)

# Plot the relationship of kms_driven with Price
st.subheader('Relationship of kms_driven with Price')
sns.relplot(x='kms_driven', y='Price', data=car, height=7, aspect=1.5)
st.pyplot()

# Plot the relationship of Fuel Type with Price
st.subheader('Relationship of Fuel Type with Price')
fig, ax = plt.subplots(figsize=(14, 7))
sns.boxplot(x='fuel_type', y='Price', data=car)
st.pyplot(fig)

# Build and train the machine learning model (same code as in your original code)

# Save the trained model
pickle.dump(pipe, open('LinearRegressionModel.pkl', 'wb'))

# Streamlit app for car price prediction
st.header('Car Price Prediction')

st.subheader('Enter Car Details for Price Prediction')
name = st.text_input('Car Name')
company = st.text_input('Company')
year = st.number_input('Year', min_value=1900, max_value=2023)
kms_driven = st.number_input('Kilometers Driven')
fuel_type = st.selectbox('Fuel Type', car['fuel_type'].unique())

input_data = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'], data=[
                          [name, company, year, kms_driven, fuel_type]])

if st.button('Predict Price'):
    model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
    predicted_price = model.predict(input_data)
    st.subheader(f'Predicted Price: ₹{predicted_price[0]:,.2f}')

# Run the Streamlit app
if __name__ == '__main__':
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_page_config(layout="wide")
    st.write('Car Price Predictor Web App')
