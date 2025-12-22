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
    plot_metrics, test_models_on_timeseries, Config, plot_training_live, average_metrics,
    plot_sensor_data_with_wear
)

# Set parameters for REINFORCE agent
episodes = Config.EPISODES
learning_rate = Config.LEARNING_RATE
gamma = Config.GAMMA
model_file = Config.REINFORCE_MODEL
ppo_model_file = Config.PPO_MODEL
reinforce_am_model_file = Config.REINFORCE_AM_MODEL

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
    st.session_state.view_mode = 'idle' # options: idle, training_rf, training_ppo, training_rf_am, evaluation, compare
if 'metrics' not in st.session_state:
    st.session_state.metrics = {
        'REINFORCE': None,
        'PPO': None,
        'REINFORCE_AM': None
    }
if 'training_log' not in st.session_state:
    st.session_state.training_log = {
        'REINFORCE': [],
        'PPO': [],
        'REINFORCE_AM': []
    }
if 'averaged_metrics' not in st.session_state:
    st.session_state.averaged_metrics = {
        'REINFORCE': None,
        'PPO': None,
        'REINFORCE_AM': None
    }
if 'sensor_data_plot' not in st.session_state:
    st.session_state.sensor_data_plot = None
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

# --- Helper Functions for Dummy Logic ---

def train_reinforce_agent(data_file: str, episodes: int = 50, save_path: str = None):
    """
    Train REINFORCE agent and store metrics and averages.
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
            model_file=save_path if save_path else model_file,
            agent_name="REINFORCE"
        )
        
        # Train
        metrics = agent.learn(total_episodes=episodes)
        
        # Store metrics and calculate averages
        st.session_state.metrics['REINFORCE'] = metrics
        st.session_state.averaged_metrics['REINFORCE'] = {
            'avg_reward': float(np.nanmean(metrics['rewards'])),
            'avg_violations': float(np.nanmean(metrics['violations'])),
            'avg_replacements': float(np.nanmean(metrics['replacements'])),
            'avg_margin': float(np.nanmean([m for m in metrics['margins'] if not np.isnan(m)]))
        }
        st.session_state.training_log['REINFORCE'].append({
            'timestamp': pd.Timestamp.now(),
            'metrics': metrics,
            'episodes': episodes
        })

        st.toast(f"Agent trained. Model saved as: {model_file}", icon="💾", duration=3)
        return agent, metrics
    
    except Exception as e:
        st.error(f"Error training REINFORCE: {str(e)}")
        return None, None


def train_ppo_agent(data_file: str, episodes: int = 50, save_path: str = None):
    """
    Train PPO agent and store metrics and averages.
    """
    try:
        # Create environment
        env = MT_Env(data_file=data_file)
        
        # Train
        metrics = train_ppo(
            env=env,
            total_episodes=episodes,
            learning_rate=learning_rate,
            gamma=gamma,
            model_file=save_path if save_path else ppo_model_file
        )
        
        # Store metrics and calculate averages
        st.session_state.metrics['PPO'] = metrics
        st.session_state.averaged_metrics['PPO'] = {
            'avg_reward': float(np.nanmean(metrics['rewards'])),
            'avg_violations': float(np.nanmean(metrics['violations'])),
            'avg_replacements': float(np.nanmean(metrics['replacements'])),
            'avg_margin': float(np.nanmean([m for m in metrics['margins'] if not np.isnan(m)]))
        }
        st.session_state.training_log['PPO'].append({
            'timestamp': pd.Timestamp.now(),
            'metrics': metrics,
            'episodes': episodes
        })

        st.toast(f"PPO agent trained. Model saved as: {ppo_model_file}", icon="💾", duration=3)
        return metrics
    
    except Exception as e:
        st.error(f"Error training PPO: {str(e)}")
        return None


def train_reinforce_am_agent(data_file: str, episodes: int = 50, save_path: str = None):
    """
    Train REINFORCE with Attention agent and store metrics and averages.
    """
    try:
        # Create attention-augmented environment
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
            learning_rate=learning_rate,
            gamma=gamma,
            model_file=save_path if save_path else reinforce_am_model_file,
            agent_name="REINFORCE_AM"
        )
        
        # Train
        metrics = agent.learn(total_episodes=episodes)
        
        # Store metrics and calculate averages
        st.session_state.metrics['REINFORCE_AM'] = metrics
        st.session_state.averaged_metrics['REINFORCE_AM'] = {
            'avg_reward': float(np.nanmean(metrics['rewards'])),
            'avg_violations': float(np.nanmean(metrics['violations'])),
            'avg_replacements': float(np.nanmean(metrics['replacements'])),
            'avg_margin': float(np.nanmean([m for m in metrics['margins'] if not np.isnan(m)]))
        }
        st.session_state.training_log['REINFORCE_AM'].append({
            'timestamp': pd.Timestamp.now(),
            'metrics': metrics,
            'episodes': episodes
        })

        st.toast(f"REINFORCE with Attention agent trained. Model saved as: {reinforce_am_model_file}", icon="💾", duration=3)
        return agent, metrics
    
    except Exception as e:
        st.error(f"Error training REINFORCE with Attention: {str(e)}")
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
    
    train_file = st.file_uploader("Select milling machine data:", type=['csv', 'txt', 'json'], key="train_uploader")
    
    if train_file:
        st.toast("Training file loaded.", icon="✅", duration=3)
        # Auto-generate sensor data visualization
        if st.session_state.uploaded_data is None or st.session_state.uploaded_data.name != train_file.name:
            try:
                df = pd.read_csv(train_file)
                st.session_state.uploaded_data = train_file
                st.session_state.sensor_data_plot = plot_sensor_data_with_wear(
                    df=df,
                    data_file_name=train_file.name,
                    smoothing=Config.PLOT_SMOOTHING_FACTOR
                )
                st.toast("Sensor data visualization generated!", icon="📊", duration=3)
            except Exception as e:
                st.warning(f"Could not generate sensor visualization: {str(e)}")
    
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
        
    # Button 1: Train PPO
    if st.button('Train PPO agent', use_container_width=True):
        st.session_state.view_mode = 'training_ppo'
    # Progress Bar container
    pbar_ppo = st.empty()

    # Button 2: Train REINFORCE
    if st.button('Train REINFORCE agent', use_container_width=True):
        st.session_state.view_mode = 'training_rf'   
    # Progress Bar container (will be used by logic below)
    pbar_rf = st.empty()

    # Button 3: Train REINFORCE with Attention
    if st.button('Train REINFORCE with Attention', use_container_width=True):
        st.session_state.view_mode = 'training_rf_am'
    # Progress Bar container
    pbar_rf_am = st.empty()
    
    st.markdown("---")
    
    # Count available metrics for comparison
    available_metrics = sum([
        st.session_state.metrics['REINFORCE'] is not None,
        st.session_state.metrics['PPO'] is not None,
        st.session_state.metrics['REINFORCE_AM'] is not None
    ])
    
    # Button 4: Compare Agents (enabled only when at least 2 metrics available)
    compare_enabled = available_metrics >= 2
    if st.button('Compare Agents', use_container_width=True, disabled=not compare_enabled):
        st.session_state.view_mode = 'compare'
    
    if not compare_enabled and available_metrics > 0:
        st.caption(f"ℹ️ Train at least 2 agents to enable comparison ({available_metrics}/3)")
    
    st.markdown("---")
    
    # Button 5: View Training Logs
    if st.button('View Training Logs', use_container_width=True):
        st.session_state.view_mode = 'training_logs'

    # --- Section 2: Agent Evaluation ---
    st.header('Agent Evaluation')

    # UPDATED: Added File Picker for Evaluation
    eval_file = st.file_uploader("Select evaluation dataset:", type=['csv', 'txt', 'json'], key="eval_uploader")

    if eval_file:
        st.toast("Evaluation file loaded!", icon="✅", duration=3)

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
        if st.session_state.sensor_data_plot is not None:
            st.markdown("### Uploaded Sensor Data Visualization")
            st.pyplot(st.session_state.sensor_data_plot)
            st.markdown("---")
            st.info("<< Training file loaded. Select an agent and start training from the sidebar.")
        else:
            st.info("<< Please select a training file and start training an agent from the sidebar.")
            st.markdown("### System Status")
            st.write("Ready to process milling machine telemetry data.")

    # --- Training View ---
    elif st.session_state.view_mode == 'training_rf':
        # st.subheader("Training REINFORCE Agent")
        
        # Train with live visualization
        agent, metrics = train_reinforce_agent(
            data_file=train_file,
            episodes=episodes,
            save_path=model_file
        )
        
        st.toast("✅ REINFORCE Training Complete. Model saved.", icon="💾", duration=3)

    elif st.session_state.view_mode == 'training_ppo':
        # st.subheader("Training PPO Agent")
        
        # Train PPO agent
        metrics = train_ppo_agent(
            data_file=train_file,
            episodes=episodes,
            save_path=ppo_model_file
        )
        
        st.toast("✅ PPO Training Complete. Model saved.", icon="💾", duration=3)

    elif st.session_state.view_mode == 'training_rf_am':
        st.subheader("Training REINFORCE with Attention Agent")
        
        # Train REINFORCE+AM agent
        agent, metrics = train_reinforce_am_agent(
            data_file=train_file,
            episodes=episodes,
            save_path=reinforce_am_model_file
        )
        
        st.toast("✅ REINFORCE+Attention Training Complete. Model saved.", icon="💾", duration=3)
    
    # --- Comparison View ---
    elif st.session_state.view_mode == 'compare':
        # st.subheader("Agent Performance Comparison")
        
        # Get available metrics (ordered: PPO, REINFORCE, REINFORCE_AM)
        available_agents = []
        if st.session_state.metrics['PPO'] is not None:
            available_agents.append('PPO')
        if st.session_state.metrics['REINFORCE'] is not None:
            available_agents.append('REINFORCE')
        if st.session_state.metrics['REINFORCE_AM'] is not None:
            available_agents.append('REINFORCE_AM')
        
        # Display names mapping
        agent_display_names = {
            'REINFORCE': 'REINFORCE',
            'PPO': 'PPO',
            'REINFORCE_AM': 'REINFORCE with Attention'
        }
        
        if len(available_agents) >= 2:
            # Display average metrics comparison with user-friendly names
            st.markdown("### Average Metrics Comparison")
            st.markdown("")
            
            # Metric display names
            metric_names = {
                'avg_reward': 'Average Reward',
                'avg_violations': 'Violation Rate',
                'avg_replacements': 'Replacement Rate',
                'avg_margin': 'Wear Margin'
            }
            
            # Create comparison columns
            cols = st.columns(len(available_agents) + 1, gap="small")
            cols[0].markdown("**Metric**")
            for i, agent in enumerate(available_agents):
                display_name = agent_display_names[agent]
                cols[i+1].markdown(f"<div style='text-align: right; margin: 0; padding: 0;'><b>{display_name}</b></div>", unsafe_allow_html=True)
            
            # Display each metric row
            for metric_key, metric_display in metric_names.items():
                cols = st.columns(len(available_agents) + 1, gap="small")
                cols[0].markdown(metric_display)
                for i, agent in enumerate(available_agents):
                    value = st.session_state.averaged_metrics[agent].get(metric_key, np.nan)
                    if np.isnan(value):
                        cols[i+1].markdown("<div style='text-align: right; margin: 0; padding: 0;'>N/A</div>", unsafe_allow_html=True)
                    else:
                        cols[i+1].markdown(f"<div style='text-align: right; margin: 0; padding: 0;'>{value:.3f}</div>", unsafe_allow_html=True)
                # Add darker horizontal line after each row
                st.markdown("<hr style='margin: 4px 0; padding: 0; border: none; border-top: 1px solid rgba(150, 150, 150, 0.5);'>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Display comparison plots
            st.markdown("#### Training Progress Comparison")
            
            # Create comparison figure for all available agents
            fig = plt.figure(figsize=(16, 6))
            
            W, H = 16, 6
            FONTSIZE_SUPER = 14
            FONTSIZE_TITLE = 11
            FONTSIZE_LABEL = 10
            FONTSIZE_TICK = 9
            BACKGROUND_COLOR = '#fafafa'
            
            metrics_config = [
                {'data': 'rewards', 'title': 'Learning Curve (Smoothed Average Reward)', 'ylabel': 'Average Reward'},
                {'data': 'violations', 'title': 'Threshold Violations (Smoothed)', 'ylabel': 'Violation Rate'},
                {'data': 'replacements', 'title': 'Replacements per Episode (Smoothed)', 'ylabel': 'Replacement Rate'},
                {'data': 'margins', 'title': 'Wear Margin at Replacement (Smoothed)', 'ylabel': 'Wear Margin'}
            ]
            
            fig, axs = plt.subplots(2, 2, figsize=(W, H))
            fig.suptitle('Agent Performance Comparison', fontsize=FONTSIZE_SUPER, color='#2C3E50')
            
            colors = {'REINFORCE': '#1f77b4', 'PPO': '#ff7f0e', 'REINFORCE_AM': '#2ca02c'}
            
            for idx, config in enumerate(metrics_config):
                ax = axs[idx // 2, idx % 2]
                data_key = config['data']
                
                for agent_name in available_agents:
                    metrics_dict = st.session_state.metrics[agent_name]
                    data = metrics_dict.get(data_key, [])
                    
                    if len(data) > 0:
                        if config['data'] == 'margins':
                            data = [m for m in data if not np.isnan(m)]
                        
                        if len(data) > 1:
                            smooth_data = pd.Series(data).rolling(window=10, min_periods=1).mean().to_numpy()
                        else:
                            smooth_data = np.array(data)
                        
                        ax.plot(smooth_data, label=agent_name, color=colors.get(agent_name, 'blue'), linewidth=2, alpha=0.8)
                
                ax.set_title(config['title'], fontsize=FONTSIZE_TITLE)
                ax.set_xlabel('Episode', fontsize=FONTSIZE_LABEL)
                ax.set_ylabel(config['ylabel'], fontsize=FONTSIZE_LABEL)
                ax.tick_params(labelsize=FONTSIZE_TICK)
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.set_facecolor(BACKGROUND_COLOR)
                ax.legend(loc='best', fontsize=9)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            st.pyplot(fig)
        else:
            st.warning("Not enough agents trained for comparison. Train at least 2 agents.")
    
    # --- Training Logs View ---
    elif st.session_state.view_mode == 'training_logs':
        st.subheader("Training Logs & History")
        
        # Display sensor data visualization as FIRST plot
        if st.session_state.sensor_data_plot is not None:
            with st.expander("📊 Input Sensor Data Visualization", expanded=True):
                st.pyplot(st.session_state.sensor_data_plot)
                st.markdown("---")
        
        agents_with_logs = [agent for agent in ['REINFORCE', 'PPO', 'REINFORCE_AM'] 
                            if st.session_state.training_log[agent]]
        
        if agents_with_logs:
            for agent_name in agents_with_logs:
                logs = st.session_state.training_log[agent_name]
                
                with st.expander(f"📊 {agent_name} Training Sessions ({len(logs)} session(s))"):
                    for i, log_entry in enumerate(logs):
                        st.write(f"**Session {i+1}** - {log_entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} ({log_entry['episodes']} episodes)")
                        
                        # Display metrics for this session
                        metrics_dict = log_entry['metrics']
                        
                        # Create visualization for this session
                        fig = plot_training_live(
                            metrics=metrics_dict,
                            episode=log_entry['episodes'],
                            total_episodes=log_entry['episodes'],
                            agent_name=agent_name,
                            window=10
                        )
                        st.pyplot(fig)
                        st.markdown("---")
        else:
            st.info("No training logs available yet. Train an agent to generate logs.")


    # --- Evaluation View ---
    elif st.session_state.view_mode == 'evaluation':
        st.subheader(f"Evaluation Results: Model {st.session_state.eval_model}")
        
        # Check if file is uploaded (optional check for the dummy logic)
        if eval_file:
            st.toast(f"Evaluated on: {eval_file.name}", icon="✅", duration=3)
        else:
            st.toast("Using default validation set", icon="ℹ️", duration=3)
        
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