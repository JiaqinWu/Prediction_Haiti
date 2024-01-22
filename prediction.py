import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import plotly.graph_objects as go

# Import Logo
image = "CGHPI.png"

# Load the dataset
df = pd.read_csv("Prediction_df.csv")

# Set up the page configuration
st.set_page_config(
    page_title="HIV Treatment Status Prediction",
    page_icon="👩‍⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Use columns for side-by-side layout
col1, col2 = st.columns([1, 6])  # Adjust the width ratio as needed

# Place the image and title in the columns
with col1:
    st.image(image, width=120)

with col2:
    st.title("HIV Treatment Status Prediction")

        
# Add sidebar
st.sidebar.title('Choose the independent variables to get the predicted treatment status!')
with st.sidebar:
    st.markdown("### Personal Information")
    sex = st.selectbox(label='Gender', options=['Female', 'Male'])
    diag = st.number_input(label="Age at Diagnosis", value=0, min_value=0)  # Ensure minimum value is 0
    facility = st.selectbox(label='Is the facility you visited the same as that you dispensed', options=['Yes', 'No'])

    st.markdown("### Dispensation Information")
    dispy = st.number_input(label="Number of years you've been getting dispensation", value=0, min_value=0)
    dispd = int(st.number_input(label="Number of days you're usually early or late to get dispensation", value=0))
    prope = st.number_input(label="Percentage of times you're early to get dispensation (%)", value=0, min_value=0, max_value=100)
    propo = st.number_input(label="Percentage of times you're on-time to get dispensation (%)", value=0, min_value=0, max_value=100)
    propl = st.number_input(label="Percentage of times you're late to get dispensation (%)", value=100 - prope - propo, min_value=0, max_value=100, disabled=True)
    avgd = st.number_input(label="Average days between your recent two dispensations", value=1, min_value=1)
    avgn = st.number_input(label="Average days to your next dispensation", value=1, min_value=1)
    yeara = st.number_input(label="Number of years you've been in Actif Status", value=0, min_value=0)

    st.markdown("### Visit Information")
    numberm = st.number_input(label="Average days between your recent two visits", value=0, min_value=0)
    numberv = int(st.number_input(label="Number of times you've visited", value=0, min_value=0))

    st.markdown("### Diagnostics Information")
    numbert = int(st.number_input(label="Number of times you've got an HIV test", value=1, min_value=1))
    test = st.selectbox(label='Have you had an HIV test within the last year', options=['Yes', 'No'])
    recent = st.selectbox(label='Recent HIV Test Result', options=['Indetectable', 'Detectable'])


# Import the scaler and model
scaler = pickle.load(open("scaler.pkl", "rb"))
model = pickle.load(open('model.pkl', 'rb'))

# Transform the value into 2 lists
lis1 = [diag, dispy, dispd, prope, propo, propl, avgd, avgn, yeara, numbert, numberm, numberv]
fem = int(sex == 'Female')
same = int(facility == 'Yes')
tes = int(test == 'Yes')
res = int(recent == 'Indetectable')
lis2 = [fem, same, tes, res]

# Define a button to trigger predictions
predict_button = st.sidebar.button("Predict")

# Define a function to plot radar plot
def create_radar(lis1):
    fig = go.Figure()

    # Add the traces
    fig.add_trace(go.Scatterpolar(
        r=lis1,  
        theta=['Age at Diagnosis', 'Dispensation Years', 'Dispensation Days Early/Late', 'Early Percentage',
               'On-time Percentage', 'Late Percentage', 'Avg Dispensation Gap', 'Avg Days to Next Dispensation',
               'Years in Actif Status', 'Number of HIV Tests', 'Avg Days Between Visits'],
        fill='toself',
        name='Patient'
    ))

    # Update the layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(lis1)]  
            )
        ),
        showlegend=True,
        autosize=True
    )

    return fig


def display_predictions(lis1, lis2, model, scaler):
    # Combine continuous and categorical variables without scaling lis2
    input_array = np.array(lis1 + lis2).reshape(1, -1)

    # Display the prediction result
    st.subheader('Treatment Status Prediction')

    # Check if the "Predict" button is clicked
    if predict_button:
        # Scale the continuous variables
        input_data_scaled = scaler.transform(input_array[:, :len(lis1)])
        # Combine the scaled continuous variables and categorical variables
        input_data_fin = np.concatenate([input_data_scaled, input_array[:, len(lis1):]], axis=1)
        # Make predictions
        prediction = model.predict(input_data_fin)

        if prediction == 1:
            st.write("<div style='font-size:30px; color:#8B0000;'>Actif</div>", unsafe_allow_html=True)
        else:
            st.write("<div style='font-size:30px; color:#8B0000;'>PIT</div>", unsafe_allow_html=True)

    else:
        # Display an empty space
        st.write(" " * 50)
    
    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")

# Main content
with st.container():
    st.title("About the App")
    st.write("This app assists in diagnosing the patients' current treatment status from their records. \
             It utilizes the XGBoost model, proven to be the most reliable with over 90% accuracy in our case. \
             The model predicts whether the patient's treatment status is 'Actif' or 'PIT' based on various features \
             such as gender, historical visit date, etc. You can update the measurements by adjusting values \
             using the sliders in the sidebar.")

    # Layout for additional content
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader('Radar Chart for Numeric Variables')
        fig = create_radar(lis1)
        # Display the radar plot
        st.plotly_chart(fig)

    with col2:
        display_predictions(lis1, lis2, model, scaler)
