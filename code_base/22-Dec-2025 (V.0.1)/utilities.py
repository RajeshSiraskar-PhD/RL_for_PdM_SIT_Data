# UTILITIES
# V.0.1: Re-baselined 30-Oct-2024
# Setting background color ax.set_facecolor('#EFEFEF')

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime

def tool_wear_data(env_type, data_file, wear_threshold, normalize=False, add_noise=False, sampling_rate=1):
    data_file_name = (data_file.split('/')[-1]).split('.')[0]
    ## Read data
    df_raw = pd.read_csv(data_file)

    df = downsample(df_raw, sampling_rate)

    # Reset index as the downsampling disturbs the index and then PPO.learn() fails. Gives a "Key error"
    df = df.reset_index(drop=True)
    n_points = len(df.index)

    # 1. Add white noise for robustness
    if add_noise:
        df['tool_wear'] = df['tool_wear'] + np.random.normal(0, 1, n_points)/add_noise

    # tool_wear = df['tool_wear']
    # plt.figure(figsize=(10, 2.5))
    # plt.plot(tool_wear, linewidth=1)
    # plt.axhline(y = wear_threshold, color = 'r', linestyle = '--', alpha=0.3) 
    # plt.title(f'Tool wear - {data_file_name} dataset')
    # plt.grid(color='lightgray', linestyle='-', linewidth=0.5)
    # plt.show()
    # fig = plt.figure()
    # save_plot = f'Tool_wear_{data_file_name}.jpg'
    # plt.savefig(save_plot)
    
    # Normalize
    if normalize:
        WEAR_MIN = df['tool_wear'].min() 
        WEAR_MAX = df['tool_wear'].max()
        WEAR_THRESHOLD_NORMALIZED = (wear_threshold-WEAR_MIN)/(WEAR_MAX-WEAR_MIN)
        df_normalized = (df-df.min())/(df.max()-df.min())

        # df_normalized['ACTION_CODE'] = np.where(df_normalized['tool_wear'] < WEAR_THRESHOLD_NORMALIZED, 0.0, 1.0)
        # print(f'Tool wear data imported ({n_points} records). WEAR_THRESHOLD_NORMALIZED: {WEAR_THRESHOLD_NORMALIZED:4.3f}')

        df_train = df_normalized.copy(deep=True)

        tool_wear = df_normalized['tool_wear']
        action_code_normalized = df_normalized['ACTION_CODE']
        action_code = df['ACTION_CODE']
        df_train['ACTION_CODE'] = df['ACTION_CODE']
    else:
        df_train = df.copy(deep=True)
        tool_wear = df['tool_wear']
        action_code = df['ACTION_CODE']

    plt.figure(figsize=(10, 2.5))
    plt.plot(tool_wear, linewidth=1)

    if normalize:
        plt.plot(action_code_normalized, linewidth=1)
        wear_threshold_return = WEAR_THRESHOLD_NORMALIZED
        plt.axhline(y = WEAR_THRESHOLD_NORMALIZED, color = 'r', linestyle = '--', alpha=0.3) 
    else:
        plt.plot(action_code, linewidth=1)
        wear_threshold_return = wear_threshold
        plt.axhline(y = wear_threshold, color = 'r', linestyle = '--', alpha=0.3) 

    plt.title(f'Tool wear - {data_file_name} dataset')
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5)
    plt.show()
    
    return tool_wear, action_code, wear_threshold_return, df_train
    
def downsample(df, sample_rate):
    # import pandas as pd
    # df = pd.read_csv(file)
    df_downsampled = df.iloc[::sample_rate, :]
    print(f'- Down-sampling. Input data records: {len(df.index)}. Sampling rate: {sample_rate}. Expected rows {round(len(df.index)/sample_rate)}.\
    Down-sampled to {len(df_downsampled.index)} rows.')
    return(df_downsampled)

def write_metrics_report(metrics, report_file, round_decimals=8):
    from pathlib import Path
    report_file = Path(report_file)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    metrics = metrics.round(round_decimals)
    metrics.to_csv(report_file, mode='a')

def store_results(file, rounds, episodes, rewards_history, ep_tool_replaced_history):
    dt = datetime.datetime.now()
    dt_d = dt.strftime('%d-%b-%Y')
    dt_t = dt.strftime('%H:%M:%S')
    df = pd.DataFrame({'Date': dt_d, 'Time': dt_t, 'Round': rounds, 'Episode': episodes, 'Rewards': rewards_history, 'Tool_replaced': ep_tool_replaced_history})
    # Append to existing training records file
    df.to_csv(file, mode='a', index=False, header=False)
    print(f'REINFORCE algorithm results saved to {file}')

def write_test_results(results, results_file):
    from csv import writer
    with open(results_file, 'a', newline='') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(results)
        f_object.close()

def plot_learning_curve(x, rewards_history, loss_history, moving_avg_n, filename):
    fig = plt.figure()
    plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111, label='1')
    ax2 = fig.add_subplot(111, label='2', frame_on=False)

    ax.plot(x, rewards_history, color='C0')
    ax.set_xlabel('Training steps', color='C0')
    ax.set_ylabel('Rewards', color='C0')
    ax.tick_params(axis='x', color='C0')
    ax.tick_params(axis='y', color='C0')

    N = len(rewards_history)
    moving_avg = np.empty(N)
    for t in range(N):
        moving_avg[t] = np.mean(loss_history[max(0, t-moving_avg_n):(t+1)])

    ax2.scatter(x, moving_avg, color='C1', s=1)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Loss', color='C1')
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors='C1')
    plt.savefig(filename)
    plt.close()
    ### plt.show()

def single_axes_plot(x, y, title='', subtitle='', x_label='', y_label='', xticks=0, threshold=0.0, filename='plot.png'):

    # Plot y
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi= 80)
    ax.plot(x, y, color='tab:blue', linewidth=2)

    # Decorations
    ax.set_xlabel(x_label, fontsize=16)
    ax.tick_params(axis='x', rotation=0, labelsize=10)
    ax.set_ylabel(y_label, color='tab:blue', fontsize=16)
    ax.tick_params(axis='y', rotation=0, labelcolor='tab:blue')
    ax.grid(alpha=.4)
    if threshold > 0.0:
        ax.axhline(y = threshold, color = 'r', linestyle = '--',  linewidth=1.0)
        ax.grid(alpha=.4)
    ax.set_xticks(np.arange(0, len(x), xticks))
    ax.set_xticklabels(x[::xticks], rotation=90, fontdict={'fontsize':10})
    plt.suptitle(title, fontsize=16, fontweight="bold", x=0.02, y=0.96, ha="left")
    plt.title(subtitle, fontsize=10, pad=10, loc="left")
    # ax.set_title(title, fontsize=18)
    # plt.title(subtitle)
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()
    ### plt.show()

def two_variable_plot(x, y1, y2, title='', subtitle='', x_label='', y1_label='', y2_label='', xticks=0, filename='plot.png'):
    # Plot Line1 (Left Y Axis)
    fig, ax = plt.subplots(1,1,figsize=(10, 4), dpi= 80)
    ax.plot(x, y1, color='tab:green', alpha=0.7, linewidth=0.5)
    ax.plot(x, y2, color='tab:blue', linewidth=2.0)

    # Decorations
    ax.set_xlabel(x_label, fontsize=16)
    ax.tick_params(axis='x', rotation=0, labelsize=10)
    ax.set_ylabel(y2_label, color='tab:blue', fontsize=16)
    ax.tick_params(axis='y', rotation=0, labelcolor='tab:blue')
    ax.grid(alpha=.4)
    ax.set_xticks(np.arange(0, len(x), xticks))
    ax.set_xticklabels(x[::xticks], rotation=90, fontdict={'fontsize':10})
    plt.suptitle(title, fontsize=16, fontweight="bold", x=0.02, y=0.96, ha="left")
    plt.title(subtitle, fontsize=10, pad=10, loc="left")
    # ax.set_title(title, fontsize=18)
    ax.legend(['Rewards', 'Moving avg.'])

    fig.tight_layout()
    plt.savefig(filename)
    plt.close()
    ### plt.show()

def two_axes_plot(x, y1, y2, title='', subtitle='', x_label='', y1_label='', y2_label='', xticks=0, file='Wear_Plot.png', threshold_org=0.0, threshold=0.0,):
    # Plot Line1 (Left Y Axis)
    fig, ax1 = plt.subplots(1,1,figsize=(10, 4), dpi= 80)
    ax1.plot(x, y1, color='tab:orange', linewidth=2)

    # Plot Line2 (Right Y Axis)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(x, y2, color='tab:blue', alpha=0.5)

    # Decorations
    # ax1 (left Y axis)
    ax1.set_xlabel(x_label, fontsize=16)
    ax1.tick_params(axis='x', rotation=0, labelsize=10)
    ax1.set_ylabel(y1_label, color='tab:red', fontsize=16)
    ax1.tick_params(axis='y', rotation=0, labelcolor='tab:red')
    ax1.grid(alpha=.4)
    if threshold > 0.0:
        ax1.axhline(y = threshold_org, color = 'r', linestyle = 'dotted',  linewidth=1.0)
        ax1.grid(alpha=.3)
        ax1.axhline(y = threshold, color = 'r', linestyle = 'dashed',  linewidth=1.0)
        ax1.grid(alpha=.4)

    # ax2 (right Y axis)
    ax2.set_ylabel(y2_label, color='tab:blue', fontsize=16)
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_xticks(np.arange(0, len(x), xticks))
    ax2.set_xticklabels(x[::xticks], rotation=90, fontdict={'fontsize':10})
    # plt.suptitle(title, fontsize=16, fontweight="bold", x=0.02, y=0.96, ha="left")
    # plt.title(subtitle, fontsize=10, pad=10, loc="left")
    ax2.set_title(title, fontsize=18)
    fig.tight_layout()
    plt.savefig(file)
    plt.close()
    ### plt.show()

def plot_error_bounds(x, y):
    import seaborn as sns
    sns.set()

    # Compute standard error
    sem = np.std(y, ddof=1) / np.sqrt(np.size(y))
    sd = np.std(y)

    plt.figure(figsize=(9, 4))
    center_line = plt.plot(x, y, 'b-')
    fill = plt.fill_between(x, y-sd, y+sd, color='b', alpha=0.2)
    plt.margins(x=0)
    plt.legend(['Rewards'])
    plt.close()
    ### plt.show()
