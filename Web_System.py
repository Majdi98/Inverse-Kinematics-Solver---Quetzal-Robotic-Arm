import numpy as np
import pandas as pd
import joblib
import streamlit as st
from Accuracy import paper_accuracy, custom_accuracy, custom_accuracy2
from scipy.spatial.transform import Rotation as R
import os
import urllib.request
from tensorflow.keras.models import load_model
import gdown
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, Dropout,
    Dense, TimeDistributed, Input, LSTM
)
# Custom objects (still used for CNN)
custom_objects_cnn = {
    'paper_accuracy': paper_accuracy,
    'custom_accuracy': custom_accuracy,
    'custom_accuracy2': custom_accuracy2,
    "Conv1D": Conv1D,
    "BatchNormalization": BatchNormalization,
    "Dropout": Dropout,
    "Dense": Dense,
    "TimeDistributed": TimeDistributed,
    "Input": Input,
}
custom_objects_lstm = {
    "paper_accuracy": paper_accuracy,
    "custom_accuracy": custom_accuracy,
    "custom_accuracy2": custom_accuracy2,
    "LSTM": LSTM,
    "Dense": Dense,
    "Dropout": Dropout,
    "TimeDistributed": TimeDistributed,
    "Input": Input
}

cnn_model_path = "models/best_cnn_model.h5"
cnn_url = "https://drive.google.com/uc?id=1Zp7lPBOBLg231FIiePNgWWypY3t6cpgX"

lstm_model_path = "models/best_lstm_model.h5"
lstm_url = "https://drive.google.com/uc?id=1sOlmkp3bxWMCcyilskZPBiTaTssAqDJe"

os.makedirs("models", exist_ok=True)

if not os.path.exists(cnn_model_path):
    with st.spinner("Downloading large model file... please wait."):
        gdown.download(cnn_url, cnn_model_path, quiet=False)
if not os.path.exists(lstm_model_path):
    with st.spinner("Downloading large model file... please wait."):
        gdown.download(lstm_url, lstm_model_path, quiet=False) 

def main():
    st.set_page_config(page_title="Quetzal IK Solver", layout="wide")

    # Initialize session values
    if 'val_px' not in st.session_state:
        st.session_state.val_px = -17.60
        st.session_state.val_py = -10.20
        st.session_state.val_pz = 12.97
        default_rot = [0.612, 0.353, 0.707, 0.499, -0.866, 0.0, 0.612, 0.353, -0.707]
        for i, val in enumerate(default_rot):
            st.session_state[f'val_r{i+1}'] = val

    def randomize_trajectory():
        st.session_state.val_px = np.random.uniform(-40.0, 40.0)
        st.session_state.val_py = np.random.uniform(-20.0, 40.0)
        st.session_state.val_pz = np.random.uniform(0.0, 60.0)
        random_rot = R.from_euler('zyx', np.random.uniform(0, 360, 3), degrees=True).as_matrix().flatten()
        for i in range(9):
            st.session_state[f'val_r{i+1}'] = random_rot[i]

    # --- Sidebar ---
    st.sidebar.title("Configuration")
    st.sidebar.subheader("1. Select DL Model")
    model_choice = st.sidebar.radio("Architecture", ["CNN", "LSTM"])

    # Load scalers
    try:
        x_scaler = joblib.load(os.path.join('scaler', 'x_scaler.pkl'))
        y_scaler = joblib.load(os.path.join('scaler', 'y_scaler.pkl'))
    except:
        st.error("Error: Scalers not found! Please train models first.")
        st.stop()
        

    if 'cnn_model' not in st.session_state:
        st.session_state['cnn_model'] = load_model(
    cnn_model_path,
    custom_objects=custom_objects_cnn,
    compile=False
)

    if 'lstm_model' not in st.session_state:
        st.session_state['lstm_model'] = load_model(
    lstm_model_path,
    custom_objects=custom_objects_lstm,
    compile=False
)
    current_model = st.session_state['cnn_model'] if model_choice == "CNN" else st.session_state['lstm_model']

    # --- Main Content ---
    st.title("Inverse Kinematics Solver - Quetzal Robotic Arm")
    st.markdown("### Deep Learning based solution (CNN & LSTM)")
    st.write("Enter the desired Position and Orientation to predict the 6 Joint Angles.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("2. Input Trajectory")
        if st.button("Generate Random Trajectory"):
            randomize_trajectory()
            st.rerun()

        st.markdown("#### Position [cm]")
        px = st.number_input("Coordinate X", -40.0, 40.0, value=st.session_state.val_px)
        py = st.number_input("Coordinate Y", -20.0, 40.0, value=st.session_state.val_py)
        pz = st.number_input("Coordinate Z", 0.0, 60.0, value=st.session_state.val_pz)

        st.markdown("#### Orientation (Rotation Matrix - 9 values)")
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            r1 = st.number_input("R11", value=st.session_state.val_r1)
            r4 = st.number_input("R21", value=st.session_state.val_r4)
            r7 = st.number_input("R31", value=st.session_state.val_r7)
        with col_r2:
            r2 = st.number_input("R12", value=st.session_state.val_r2)
            r5 = st.number_input("R22", value=st.session_state.val_r5)
            r8 = st.number_input("R32", value=st.session_state.val_r8)
        with col_r3:
            r3 = st.number_input("R13", value=st.session_state.val_r3)
            r6 = st.number_input("R23", value=st.session_state.val_r6)
            r9 = st.number_input("R33", value=st.session_state.val_r9)

        run_pred = st.button("Predicted Inverse Kinematics", type="primary")

    with col2:
        st.subheader("3. Results & Visualization")

        if run_pred:
            is_valid = (-40.0 <= px <= 40.0) and (-20.0 <= py <= 40.0) and (0.0 <= pz <= 60.0)
            if not is_valid:
                st.error("One or more coordinates are outside the allowed workspace!")
                st.info("**Allowed Range:** X:±40, Y:[-20, 40], Z:[0, 60]")
                st.stop()  

            rot_values = np.array([r1, r2, r3, r4, r5, r6, r7, r8, r9])
            input_vector = np.concatenate(([px, py, pz], rot_values))
            input_sequence = np.tile(input_vector, (4, 1)).reshape(1, 4, 12)
            input_scaled = x_scaler.transform(input_sequence.reshape(-1, 12)).reshape(1, 4, 12)

            with st.spinner(f"Running inference with {model_choice}..."):
                if model_choice == "CNN":
                    pred_angles_scaled = st.session_state['cnn_model'].predict(input_scaled)

                else:  
                    pred_angles_scaled = st.session_state['lstm_model'].predict(input_scaled)


                pred_angles = y_scaler.inverse_transform(pred_angles_scaled.reshape(-1, 6)).reshape(1, 4, 6)

            joints = ["q1", "q2", "q3", "q4", "q5", "q6"]
            pred_angles = pred_angles[:, 0, :]
            df_res = pd.DataFrame(pred_angles, columns=joints)
            st.dataframe(df_res)
            st.bar_chart(df_res.T)
        else:
            st.info("Waiting for prediction...")

main()








