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

# Set parameters for REINFORCE agent
learning_rate = Config.LEARNING_RATE
gamma = Config.GAMMA
model_file = Config.REINFORCE_MODEL

# --- Page Configuration ---
st.set_page_config(
    page_title="RL for Predictive Maintenance",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Theming ---
# #d2e3fa, #87CDEE #cad3db
st.markdown("""
    <style>
        /* Main App Background (Right Column) - Very Light Grey */
        .stApp {
            background-color: #F0F2F6;
        }
        
        /* Sidebar Background (Left Column) - Pastel Blue */
        [data-testid="stSidebar"] {
            background-color: #a4d4eb;
            border-right: 1px solid #d1e7fbad;
        }

        /* Adjusting text colors */
        h1 {
            color: #2C3E50;
            font-size: 28px !important;
            margin-top: -2rem !important;
            margin-bottom: -0.5rem !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }
        
        hr {
            margin-top: 0.5rem !important;
            margin-bottom: 1rem !important;
        }
        h2 {
            color: #2C3E50;
            font-size: 22px !important;
        }
        h3 {
            color: #2C3E50;
            font-size: 18px !important;
        }
        
        /* Custom button styling */
        .stButton > button {
            width: 100%;
            background-color: #F0F2F6;
            border: 1px solid #B0C4DE;
        }
        
        .stButton > button:hover {
            background-color: #e1e6eb;
            border: 1px solid #A9C5E0;
        }
    </style>
""", unsafe_allow_html=True)

# --- Session State Management ---
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'idle' # options: idle, training_rf, training_rf_am, evaluation
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}

# --- Helper Functions for Dummy Logic ---

def train_reinforce_agent(data_file: str, episodes: int = 50, save_path: str = None):
    """
    Train REINFORCE agent.
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
            learning_rate=learning_rate,
            gamma=gamma,
            model_file=save_path if save_path else model_file
        )
        
        # Train
        metrics = agent.learn(total_episodes=episodes)

        st.toast(f"Agent trained. Model saved as: {model_file}", icon="💾")
        return agent, metrics
    
    except Exception as e:
        st.error(f"Error training REINFORCE: {str(e)}")
        return None, None


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
    # --- Section 1: Agent Training ---
    st.header('Agent Training')
    
    train_file = st.file_uploader("Select milling machine training data:", type=['csv', 'txt', 'json'], key="train_uploader")
    
    if train_file:
        st.success("Training file loaded.")
    
    # Hyperparameters
    st.subheader("Hyperparameters")
    
    episodes = st.number_input(
        "Episodes",
        min_value=1,
        value=Config.EPISODES,
        step=10
    )
    
    learning_rate = st.number_input(
        "Learning Rate (α)",
        min_value=1e-5,
        max_value=1.0,
        value=Config.LEARNING_RATE,
        format="%.5f"
    )
    
    gamma = st.number_input(
        "Discount Factor (γ)",
        min_value=0.0,
        max_value=0.9999,
        value=Config.GAMMA,
        format="%.4f"
    )
        
    # Button 1: Train REINFORCE
    if st.button('Train REINFORCE agent', use_container_width=True):
        st.session_state.view_mode = 'training_rf'   
    # Progress Bar container (will be used by logic below)
    pbar_rf = st.empty()

    # Button 2: Train REINFORCE with Attention
    if st.button('Train REINFORCE with Attention', use_container_width=True):
        st.session_state.view_mode = 'training_rf_am'
    # Progress Bar container
    pbar_rf_am = st.empty()   
    st.markdown("---")

    # --- Section 2: Agent Evaluation ---
    st.header('Agent Evaluation')

    # UPDATED: Added File Picker for Evaluation
    eval_file = st.file_uploader("Select evaluation dataset:", type=['csv', 'txt', 'json'], key="eval_uploader")

    if eval_file:
        st.success("Evaluation file loaded!")

    model_options = ["RF", "RF_AM"]
    selected_model = st.selectbox("Select model to evaluate:", model_options)
    
    if st.button('Evaluate Saved Models', use_container_width=True):
        st.session_state.view_mode = 'evaluation'
        st.session_state.eval_model = selected_model

# ==========================================
# RIGHT PANEL (MAIN AREA)
# ==========================================

st.title('Reinforcement Learning for Predictive Maintenance')
st.markdown("---")

# Container for the dynamic right column content
main_container = st.container()

with main_container:
    if st.session_state.view_mode == 'idle':
        st.info("<< Please select a training file and start training an agent from the sidebar.")
        st.markdown("### System Status")
        st.write("Ready to process milling machine telemetry data.")

    # --- Training View ---
    elif st.session_state.view_mode == 'training_rf':
        # st.subheader("Training a REINFORCE Agent")
        # st.markdown("Watch the model train in real-time...\n---")
        
        # Train with live visualization - plots update automatically!
        agent, metrics = train_reinforce_agent(
            data_file=train_file,
            episodes=episodes,
            save_path=model_file
        )
        
        st.success("✅ Training Complete. Model saved.")

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