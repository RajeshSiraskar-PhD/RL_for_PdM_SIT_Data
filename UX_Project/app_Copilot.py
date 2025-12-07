# --------------------------------------------------------------------------------------'
# UX_Project/app_Copilot.py
# --------------------------------------------------------------------------------------

import streamlit as st
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pandas as pd

# Page config
st.set_page_config(page_title="Reinforcement Learning for Predictive Maintenance - V. COP-1 ", layout="wide")

# Custom CSS for column background colors
st.markdown(
    """
    <style>
    .left-col {
        background-color: #cce7ff; /* pastel blue */
        padding: 20px;
        border-radius: 8px;
    }
    .right-col {
        background-color: #f5f5f5; /* very light grey */
        padding: 20px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create two columns
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="left-col">', unsafe_allow_html=True)
    st.title("Reinforcement Learning for Predictive Maintenance")

    # Section 1: Agent Training
    st.header("Agent Training")
    uploaded_file = st.file_uploader("Select milling machine training data:", type=["csv", "xlsx"])

    if st.button("Train REINFORCE agent"):
        st.info("Training REINFORCE agent...")
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.02)
            progress.progress(i + 1)
        # Dummy save model
        st.success("Model 'RF' trained and saved!")

        # Show training plots in right column
        with col2:
            st.markdown('<div class="right-col">', unsafe_allow_html=True)
            st.subheader("Training Plots - REINFORCE")
            fig, ax = plt.subplots()
            ax.plot(np.random.randn(100).cumsum(), label="Reward")
            ax.set_title("Training Reward Curve")
            ax.legend()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Train REINFORCE with Attention"):
        st.info("Training REINFORCE with Attention...")
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.02)
            progress.progress(i + 1)
        # Dummy save model
        st.success("Model 'RF_AM' trained and saved!")

        # Show training plots in right column
        with col2:
            st.markdown('<div class="right-col">', unsafe_allow_html=True)
            st.subheader("Training Plots - REINFORCE + Attention")
            fig, ax = plt.subplots()
            ax.plot(np.random.randn(100).cumsum(), color="orange", label="Reward")
            ax.set_title("Training Reward Curve (Attention)")
            ax.legend()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

    # Section 2: Agent Evaluation
    st.header("Agent Evaluation")
    if st.button("Evaluate Saved Models"):
        st.info("Evaluating models...")

        # Dummy evaluation metrics
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        report = classification_report(y_true, y_pred, output_dict=True)

        with col2:
            st.markdown('<div class="right-col">', unsafe_allow_html=True)
            st.subheader("Evaluation Results")

            # Show metrics table
            metrics_df = pd.DataFrame(report).transpose()
            st.dataframe(metrics_df)

            # Show plots
            st.subheader("Evaluation Plots")
            fig, ax = plt.subplots()
            ax.bar(["Precision", "Recall", "F1-score"], 
                   [report['weighted avg']['precision'], 
                    report['weighted avg']['recall'], 
                    report['weighted avg']['f1-score']])
            ax.set_ylim(0, 1)
            st.pyplot(fig)

            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)