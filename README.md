# Haiti HIV Treatment Status Prediction App

## Overview

The Haiti HIV Treatment Status Prediction app is a machine learning-powered tool designed to assist medical professionals in diagnosing the treatment status of HIV patients in Haiti. Using a set of independent variables and machine-learning models, the app predicts whether a patient is in Actif (Active) or PIT (Out of Care) status. It provides a visual representation of the input data using a radar chart and displays the predicted treatment status. The app can be used by manually inputting the independent variables.

A live version of the application can be found on [Streamlit Community Cloud](https://haiti-prediction-cghpi.streamlit.app/). 

## Installation

To run the App locally, you will need to have Python 3.6 or higher installed. Then, you can install the required packages by running:

```bash
pip install -r requirements.txt
```

This will install all the necessary dependencies, including Streamlit, plotly, and scikit-learn.

## Usage
To start the app, simply run the following command:

```bash
streamlit run app.py
```

This will launch the app in your default web browser. Input patients' independent variables to predict the patient's treatment status.
