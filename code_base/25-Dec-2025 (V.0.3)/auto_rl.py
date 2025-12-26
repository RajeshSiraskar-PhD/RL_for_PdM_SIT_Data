import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from rl_pdm_module import (
    Config, MT_Env, AM_Env, PolicyNetwork, REINFORCE, train_ppo, plot_training_live
)

def run_auto_rl(training_file, episodes):
    # Ensure directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('saved_plots', exist_ok=True)
    
    data_file_name = os.path.basename(training_file)
    date_str = datetime.now().strftime("%d%m%Y")
    
    # Define models to train
    agent_types = ['PPO', 'REINFORCE', 'REINFORCE_AM']
    
    # Results collection
    all_results = []
    
    for agent_type in agent_types:
        print(f"\n{'='*60}")
        print(f"TRAINING AGENT: {agent_type}")
        print(f"{'='*60}")
        
        # Determine environment and training function
        if agent_type == 'REINFORCE_AM':
            env = AM_Env(data_file=training_file)
        else:
            env = MT_Env(data_file=training_file)
            
        model_path = Config.get_model_path(agent_type, episodes)
        metrics = None
        
        if agent_type == 'PPO':
            metrics = train_ppo(
                env=env,
                total_episodes=episodes,
                learning_rate=Config.LEARNING_RATE,
                gamma=Config.GAMMA,
                model_file=model_path,
                data_file_name=data_file_name
            )
        else:
            # For REINFORCE and REINFORCE_AM
            policy = PolicyNetwork(
                input_dim=env.observation_space.shape[0],
                output_dim=env.action_space.n
            )
            agent = REINFORCE(
                policy=policy,
                env=env,
                learning_rate=Config.LEARNING_RATE,
                gamma=Config.GAMMA,
                model_file=model_path,
                agent_name=agent_type,
                data_file_name=data_file_name
            )
            metrics = agent.learn(total_episodes=episodes)
            
        if metrics:
            # 1. Save training plot
            fig = plot_training_live(
                metrics=metrics,
                episode=episodes,
                total_episodes=episodes,
                agent_name=agent_type,
                data_file_name=data_file_name,
                window=5
            )
            
            # Format: 'model-name-training-file-name-episodes-ddmmyyy.png'
            plot_base_name = f"{agent_type}-{data_file_name.replace('.csv', '')}-{episodes}-{date_str}.png"
            plot_path = os.path.join('saved_plots', plot_base_name)
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"✅ Plot saved: {plot_path}")
            
            # 2. Collect average metrics
            avg_metrics = {
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Agent': agent_type,
                'File': data_file_name,
                'Episodes': episodes,
                'Avg_Reward': np.mean(metrics['rewards']),
                'Avg_Violations': np.mean(metrics['violations']),
                'Avg_Replacements': np.mean(metrics['replacements']),
                'Avg_Margin': np.nanmean([m for m in metrics['margins'] if not np.isnan(m)])
            }
            all_results.append(avg_metrics)
            print(f"✅ Training results collected for {agent_type}")

    # 3. Append results to training_results.csv
    if all_results:
        results_df = pd.DataFrame(all_results)
        csv_file = 'training_results.csv'
        
        if os.path.exists(csv_file):
            results_df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            results_df.to_csv(csv_file, mode='w', header=True, index=False)
            
        print(f"\n✅ All results appended to {csv_file}")
    
    print("\n--- Automation Script Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automated RL training for Predictive Maintenance.')
    parser.add_argument('training_file', type=str, help='Path to the training CSV file')
    parser.add_argument('--episodes', type=int, default=20, help='Number of episodes for training (default: 20)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.training_file):
        print(f"Error: File {args.training_file} not found.")
        sys.exit(1)
        
    run_auto_rl(args.training_file, args.episodes)
