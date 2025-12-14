"""
Example Usage Guide for RL_PdM Module in Streamlit App

This file demonstrates how to import and use the rl_pdm_module.py
in your Streamlit application (app.py).
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Import the RL module
from rl_pdm_module import (
    Config,
    MT_Env, AM_Env,
    PolicyNetwork, REINFORCE,
    train_ppo,
    plot_metrics, plot_single_metrics,
    average_metrics, smooth,
    test_models_on_timeseries
)

# ==========================================
# EXAMPLE 1: TRAIN REINFORCE MODEL
# ==========================================
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

# ==========================================
# EXAMPLE 2: TRAIN PPO MODEL
# ==========================================
def train_ppo_example(data_file: str, episodes: int = 50, save_path: str = None):
    """
    Example function to train PPO agent.
    Can be called from Streamlit UI.
    """
    try:
        # Create environment
        env = MT_Env(data_file=data_file)
        
        # Train
        metrics = train_ppo(
            env=env,
            total_episodes=episodes,
            learning_rate=Config.LEARNING_RATE,
            gamma=Config.GAMMA,
            model_file=save_path if save_path else Config.PPO_MODEL
        )
        
        return metrics
    
    except Exception as e:
        st.error(f"Error training PPO: {str(e)}")
        return None

# ==========================================
# EXAMPLE 3: TRAIN REINFORCE WITH ATTENTION
# ==========================================
def train_reinforce_attention_example(data_file: str, episodes: int = 50, save_path: str = None):
    """
    Example function to train REINFORCE with Attention mechanism.
    Can be called from Streamlit UI.
    """
    try:
        # Create attention environment
        env = AM_Env(data_file=data_file)
        
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
            model_file=save_path if save_path else Config.REINFORCE_AM_MODEL
        )
        
        # Train
        metrics = agent.learn(total_episodes=episodes)
        
        return agent, metrics
    
    except Exception as e:
        st.error(f"Error training REINFORCE+Attention: {str(e)}")
        return None, None

# ==========================================
# EXAMPLE 4: PLOT TRAINING METRICS
# ==========================================
def plot_training_example(metrics_rf, metrics_ppo):
    """
    Example to plot training metrics comparison.
    """
    try:
        fig = plot_metrics(
            metrics_a=metrics_rf,
            metrics_b=metrics_ppo,
            name_a="REINFORCE",
            name_b="PPO",
            window=Config.SMOOTH_WINDOW,
            mode="COMBINED"
        )
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting metrics: {str(e)}")

def plot_single_agent_metrics(metrics, agent_name: str):
    """
    Example to plot metrics for single agent.
    """
    try:
        fig = plot_single_metrics(
            metrics=metrics,
            name=agent_name,
            window=Config.SMOOTH_WINDOW
        )
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting metrics: {str(e)}")

# ==========================================
# EXAMPLE 5: DISPLAY RESULTS TABLE
# ==========================================
def display_metrics_table(metrics_rf, metrics_ppo):
    """
    Example to display average metrics in a table.
    """
    try:
        avg_metrics = average_metrics(
            metrics_a=metrics_rf,
            metrics_b=metrics_ppo,
            name_a="REINFORCE",
            name_b="PPO",
            leave_first_N=10  # Skip first 10 episodes for steady-state
        )
        
        # Convert to DataFrame for display
        df_results = pd.DataFrame.from_dict(avg_metrics, orient="index")
        st.dataframe(df_results.round(4), use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying metrics: {str(e)}")

# ==========================================
# EXAMPLE 6: EVALUATE ON TEST DATA
# ==========================================
def evaluate_on_test_data(test_file: str):
    """
    Example to evaluate all three models on test data.
    """
    try:
        df_results = test_models_on_timeseries(
            test_file=test_file,
            wear_threshold=Config.WEAR_THRESHOLD,
            ppo_model_file=Config.PPO_MODEL,
            reinforce_model_file=Config.REINFORCE_MODEL,
            reinforce_am_model_file=Config.REINFORCE_AM_MODEL
        )
        
        # Display results
        st.subheader("Model Evaluation Results")
        st.dataframe(df_results.round(4), use_container_width=True)
        
        # Download results
        csv = df_results.to_csv(index=True)
        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name="model_evaluation_results.csv",
            mime="text/csv"
        )
        
        return df_results
    
    except Exception as e:
        st.error(f"Error evaluating models: {str(e)}")
        return None

# ==========================================
# INTEGRATION WITH STREAMLIT APP
# ==========================================
"""
Here's how to integrate these functions into your app.py:

# In app.py sidebar:
st.sidebar.header('Agent Training')
if st.sidebar.button('Train REINFORCE'):
    st.session_state.metrics_rf = train_reinforce_example(
        data_file='data/PROC_9.csv',
        episodes=50
    )

if st.sidebar.button('Train PPO'):
    st.session_state.metrics_ppo = train_ppo_example(
        data_file='data/PROC_9.csv',
        episodes=50
    )

if st.sidebar.button('Train REINFORCE+Attention'):
    st.session_state.metrics_rf_am = train_reinforce_attention_example(
        data_file='data/PROC_9.csv',
        episodes=50
    )

# In app.py main area:
if 'metrics_rf' in st.session_state and 'metrics_ppo' in st.session_state:
    st.header('Training Results')
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Metrics Comparison')
        display_metrics_table(
            st.session_state.metrics_rf,
            st.session_state.metrics_ppo
        )
    
    with col2:
        st.subheader('Training Plots')
        plot_training_example(
            st.session_state.metrics_rf,
            st.session_state.metrics_ppo
        )

# For evaluation:
test_file = st.file_uploader('Select test file')
if test_file and st.button('Evaluate'):
    evaluate_on_test_data(test_file)
"""

if __name__ == "__main__":
    st.title("RL for Predictive Maintenance - Module Test")
    
    st.info("""
    This is a test file showing how to use the rl_pdm_module.py in Streamlit.
    See the code above for integration examples.
    """)
    
    # Test Config loading
    st.subheader("Configuration")
    col1, col2, col3 = st.columns(3)
    col1.metric("Learning Rate", Config.LEARNING_RATE)
    col2.metric("Discount Factor", Config.GAMMA)
    col3.metric("Episodes", Config.EPISODES)
