# ------------------------------------------
# Gemini version
# ------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from rl_pdm_module import (
    train_ppo, REINFORCE, PolicyNetwork, MT_Env, AM_Env,
    plot_metrics, test_models_on_timeseries, Config
)

# --- Page Configuration ---
st.set_page_config(
    page_title="RL for Predictive Maintenance",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Theming ---
st.markdown("""
    <style>
        /* Main App Background (Right Column) - Very Light Grey */
        .stApp {
            background-color: #F0F2F6;
        }
        
        /* Sidebar Background (Left Column) - Pastel Blue */
        [data-testid="stSidebar"] {
            background-color: #E6F3FF;
            border-right: 1px solid #d1e7fbad;
        }

        /* Adjusting text colors */
        h1, h2, h3 {
            color: #2C3E50;
        }
        
        /* Custom button styling */
        .stButton > button {
            width: 100%;
            background-color: #FFFFFF;
            border: 1px solid #B0C4DE;
        }
    </style>
""", unsafe_allow_html=True)

# --- Session State Management ---
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'idle' # options: idle, training_rf, training_rf_am, evaluation
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}

# --- Helper Functions for Dummy Logic ---

def train_reinforce_example(data_file: str, episodes: int = 50, save_path: str = None):
    """
    Example function to train REINFORCE agent.
    Can be called from Streamlit UI.
    """
    try:
        # Create environment
        env = MT_Env(data_file=data_file)
        
        # Create policy network
        policy = PolicyNetwork(
            input_dim=env.observation_space.shape[0],
            output_dim=env.action_space.n
        )
        
        # Create agent
        agent = REINFORCE(
            policy=policy,
            env=env,
            learning_rate=Config.LEARNING_RATE,
            gamma=Config.GAMMA,
            model_file=save_path if save_path else Config.REINFORCE_MODEL
        )
        
        # Train
        metrics = agent.learn(total_episodes=episodes)
        
        return agent, metrics
    
    except Exception as e:
        st.error(f"Error training REINFORCE: {str(e)}")
        return None, None


def simulate_training(model_name, plot_placeholder, progress_bar):
    """Simulates training loop with live plotting."""
    loss_values = []
    reward_values = []
    
    steps = 50
    for i in range(steps):
        # Simulate calculations
        time.sleep(0.05) 
        loss = np.exp(-i/10) + np.random.normal(0, 0.05)
        reward = np.log(i+1) + np.random.normal(0, 0.1)
        
        loss_values.append(loss)
        reward_values.append(reward)
        
        # Update Progress
        progress = (i + 1) / steps
        progress_bar.progress(progress)
        
        # Update Plots in Right Column
        with plot_placeholder.container():
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Training Loss")
                st.line_chart(loss_values, height=250)
            with col2:
                st.subheader("Cumulative Reward")
                st.line_chart(reward_values, height=250)
    
    # Save "Dummy" Model
    st.toast(f"Model saved as: {model_name}", icon="💾")
    return {"loss": loss_values, "reward": reward_values}

def generate_eval_metrics():
    """Generates dummy evaluation metrics."""
    return {
        "Accuracy": f"{np.random.randint(85, 98)}%",
        "Precision": f"{np.random.randint(80, 95)}%",
        "Recall": f"{np.random.randint(80, 95)}%",
        "F1 Score": f"{np.random.uniform(0.8, 0.95):.2f}"
    }

# ==========================================
# LEFT PANEL (SIDEBAR)
# ==========================================
with st.sidebar:
    st.title('Reinforcement Learning for Predictive Maintenance')
    
    st.markdown("---")
    
    # --- Section 1: Agent Training ---
    st.header('1. Agent Training')
    
    train_file = st.file_uploader("Select milling machine training data:", type=['csv', 'txt', 'json'], key="train_uploader")
    
    if train_file:
        st.success("Training file loaded!")
    
    st.write("###") # Spacer
    
    # Button 1: Train REINFORCE
    if st.button('Train REINFORCE agent'):
        st.session_state.view_mode = 'training_rf'
        
    # Progress Bar container (will be used by logic below)
    pbar_rf = st.empty()

    st.write("###") # Spacer

    # Button 2: Train REINFORCE with Attention
    if st.button('Train REINFORCE with Attention'):
        st.session_state.view_mode = 'training_rf_am'
    
    # Progress Bar container
    pbar_rf_am = st.empty()
    
    st.markdown("---")

    # --- Section 2: Agent Evaluation ---
    st.header('2. Agent Evaluation')

    # UPDATED: Added File Picker for Evaluation
    eval_file = st.file_uploader("Select evaluation dataset:", type=['csv', 'txt', 'json'], key="eval_uploader")

    if eval_file:
        st.success("Evaluation file loaded!")

    st.write("###") # Spacer
    
    model_options = ["RF", "RF_AM"]
    selected_model = st.selectbox("Select model to evaluate:", model_options)
    
    if st.button('Evaluate Saved Models'):
        st.session_state.view_mode = 'evaluation'
        st.session_state.eval_model = selected_model

# ==========================================
# RIGHT PANEL (MAIN AREA)
# ==========================================

# Container for the dynamic right column content
main_container = st.container()

with main_container:
    if st.session_state.view_mode == 'idle':
        st.info("<< Please select a training file and start training an agent from the sidebar.")
        st.markdown("### System Status")
        st.write("Ready to process milling machine telemetry data.")

    # --- Training View ---
    elif st.session_state.view_mode == 'training_rf':
        st.subheader("Training Progress: REINFORCE Agent")
        st.markdown("Wait for the training loop to complete...")
        plot_area = st.empty()

        # $$$ Trigger the REINFORCE training function
        train_reinforce_example(data_file=train_file, episodes=Config.EPISODES, save_path=Config.REINFORCE_MODEL)
        
        st.success("Training Complete. Model 'RF' saved.")

    elif st.session_state.view_mode == 'training_rf_am':
        st.subheader("Training Progress: REINFORCE with Attention")
        st.markdown("Wait for the training loop to complete...")
        plot_area = st.empty()
        # Trigger simulation
        simulate_training('RF_AM', plot_area, pbar_rf_am)
        st.success("Training Complete. Model 'RF_AM' saved.")

    # --- Evaluation View ---
    elif st.session_state.view_mode == 'evaluation':
        st.subheader(f"Evaluation Results: Model {st.session_state.eval_model}")
        
        # Check if file is uploaded (optional check for the dummy logic)
        if not eval_file:
            st.warning("Note: Using default validation set (No evaluation file uploaded).")
        else:
            st.success(f"Evaluated on: {eval_file.name}")
        
        # 1. Statistical Results (Top Area)
        metrics = generate_eval_metrics()
        
        st.markdown("#### Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", metrics["Accuracy"])
        col2.metric("Precision", metrics["Precision"])
        col3.metric("Recall", metrics["Recall"])
        col4.metric("F1 Score", metrics["F1 Score"])
        
        st.markdown("---")
        
        # 2. Plots (Bottom Area)
        st.markdown("#### Diagnostic Plots")
        
        col_plot1, col_plot2 = st.columns(2)
        
        with col_plot1:
            st.markdown("**Confusion Matrix**")
            # Mock Confusion Matrix Data
            heatmap_data = np.random.rand(5, 5)
            fig, ax = plt.subplots()
            cax = ax.imshow(heatmap_data, cmap='Blues')
            fig.colorbar(cax)
            st.pyplot(fig)
            
        with col_plot2:
            st.markdown("**Maintenance Prediction Probability**")
            # Mock Time Series Data
            chart_data = pd.DataFrame(
                np.random.randn(20, 3),
                columns=['Normal', 'Warning', 'Critical']
            )
            st.area_chart(chart_data)