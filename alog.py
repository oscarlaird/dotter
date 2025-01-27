#%%
!pip install seaborn matplotlib --break-system-packages
#%%
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

# Set style for all plots
plt.style.use('seaborn-v0_8')  # Use the valid style name for seaborn
sns.set_context("notebook", font_scale=1.2)
sns.set_palette("husl")

def load_data(filename):
    """Load and parse log file data"""
    objects = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                objects.append(data)
            except json.JSONDecodeError:
                continue
    return objects

def get_user_counts(objects):
    """Count entries per username and print sorted results with total time"""
    username_counts = {}
    username_times = {}
    for data in objects:
        username = data.get('username', 'unknown').strip()
        username_counts[username] = username_counts.get(username, 0) + 1
        username_times[username] = username_times.get(username, 0) + data.get('time_elapsed', 0)
    
    # Create DataFrame for prettier display
    df = pd.DataFrame({
        'Username': list(username_counts.keys()),
        'Entries': list(username_counts.values()),
        'Total Time (s)': [username_times[u] for u in username_counts.keys()]
    }).sort_values('Entries', ascending=False)
    
    print("\nEntries and time per username:")
    print(df.to_string(index=False))
    return username_counts

def get_user_entries(objects, username):
    """Filter entries for specific username"""
    return [data for data in objects if data.get('username', '').strip() == username or data.get('username', '').strip() == 'null']

def calculate_metrics(entries):
    """Calculate WPM, scan times, delay STDs and outlier rates"""
    wpm_values = [len(entry['best_val'][:-1])/entry['time_elapsed']*60/5 for entry in entries]
    
    scan_times = [entry['time_elapsed']/(len(entry['delay_pairs'])-1) - entry['delay_pairs'][0]['period']/2 
                 for entry in entries]
    
    outlier_rates = []
    for entry in entries:
        delays = [pair['delay'] for pair in entry['delay_pairs']]
        outliers = [d for d in delays if abs(d) > entry['delay_pairs'][0]['period']*0.25]
        outlier_rate = 2*len(outliers) / len(delays)
        outlier_rates.append(outlier_rate)
        
    delay_stds = [np.std([pair['delay'] for pair in entry['delay_pairs'] if abs(pair['delay']) <0.4]) 
                 for entry in entries]

    cum_time_elapsed = np.array([sum(entry['time_elapsed'] for entry in entries[:i+1]) 
                                for i in range(len(entries))])
    cum_time_elapsed *= 7200 / cum_time_elapsed[-1]
    
    return wpm_values, scan_times, delay_stds, outlier_rates, cum_time_elapsed

def calculate_rolling_average(values, alpha=0.05):
    """Calculate exponential moving average"""
    rolling_avg = [values[0]]
    ema = values[0]
    for val in values[1:]:
        ema = alpha * val + (1 - alpha) * ema
        rolling_avg.append(ema)
    return rolling_avg

def plot_metric(ax, x, y, rolling_avg, title, ylabel, username):
    """Plot a single metric with rolling average and trend line"""
    # Create scatter plot with better styling
    sns.scatterplot(x=x/60, y=y, alpha=0.3, label=f'{username} Raw', ax=ax)
    
    # Plot rolling average with confidence interval
    sns.lineplot(x=x/60, y=rolling_avg, color='red', linewidth=2, 
                label=f'{username} Rolling Avg', ax=ax)
    
    # Add trend line with confidence interval
    sns.regplot(x=x/60, y=y, scatter=False, color='green', 
                line_kws={'linestyle': '--', 'alpha': 0.8}, ax=ax)
    
    ax.set_xlabel('Time Elapsed (minutes)')
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=15)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)

def plot_user_metrics(entries, username):
    """Plot all metrics for a single user"""
    wpm_values, scan_times, delay_stds, outlier_rates, cum_time_elapsed = calculate_metrics(entries)
    
    # Create figure with custom style
    plt.rcParams['figure.figsize'] = [15, 10]
    fig, axes = plt.subplots(2, 2)
    fig.suptitle(f'Performance Metrics for User "{username}"', 
                 fontsize=16, y=1.02)
    
    metrics = [
        (wpm_values, "WPM vs Time", "Words Per Minute"),
        (scan_times, "Avg Scan Time vs Time", "Avg Scan Time (seconds)"),
        (delay_stds, "Delay Standard Deviation vs Time", "Std Dev of Delays (seconds)"),
        (outlier_rates, "Outlier Rate vs Time", "Outlier Rate")
    ]
    
    for (metric, title, ylabel), ax in zip(metrics, axes.flat):
        rolling_avg = calculate_rolling_average(metric)
        plot_metric(ax, cum_time_elapsed, metric, rolling_avg, title, ylabel, username)
    
    plt.tight_layout()
    sns.despine()
    plt.show()

def plot_comparative_metrics(objects, usernames):
    """Plot metrics comparing multiple users"""
    # Set up the figure with a clean, modern style
    plt.rcParams['figure.figsize'] = [15, 10]
    fig, axes = plt.subplots(2, 2)
    fig.suptitle('Comparative Performance Metrics', 
                 fontsize=16, y=1.02)
    
    metric_names = [
        "WPM vs Time", 
        "Avg Scan Time vs Time",
        "Delay Standard Deviation vs Time",
        "Outlier Rate vs Time"
    ]
    ylabels = [
        "Words Per Minute",
        "Avg Scan Time (seconds)", 
        "Std Dev of Delays (seconds)",
        "Outlier Rate"
    ]
    
    # Store rolling averages for each user and metric
    all_rolling_avgs = {i: [] for i in range(4)}
    all_times = []
    
    # Create color palette for users
    colors = sns.color_palette("husl", n_colors=len(usernames))
    
    for idx, username in enumerate(usernames):
        entries = get_user_entries(objects, username)
        metrics = calculate_metrics(entries)
        all_times.append(metrics[4])
        
        for i, (metric, ax) in enumerate(zip(metrics[:4], axes.flat)):
            rolling_avg = calculate_rolling_average(metric)
            sns.scatterplot(x=metrics[4]/60, y=metric, alpha=0.3, 
                          color=colors[idx], label=f'{username}', ax=ax)
            sns.lineplot(x=metrics[4]/60, y=rolling_avg, alpha=0.6, 
                        color=colors[idx], ax=ax)
            all_rolling_avgs[i].append(rolling_avg)
            
            ax.set_title(metric_names[i], pad=15)
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel(ylabels[i])
    
    # Calculate and plot mean trends
    min_len = min(len(t) for t in all_times)
    common_time = np.linspace(0, min_len, min_len)
    
    for i, ax in enumerate(axes.flat):
        interpolated_avgs = []
        for j, ra in enumerate(all_rolling_avgs[i]):
            interp_func = np.interp(common_time, np.arange(len(ra)), ra)
            interpolated_avgs.append(interp_func)
        
        mean_rolling_avg = np.mean(interpolated_avgs, axis=0)
        sns.lineplot(x=np.arange(len(mean_rolling_avg))*120/len(mean_rolling_avg), 
                    y=mean_rolling_avg, color='black', linewidth=3, 
                    label='Group Average', ax=ax)
        
        # Set y-axis limits
        ylims = [(0, 10), (0, 2.4), (0, 0.25), (0, 0.125)]
        ax.set_ylim(*ylims[i])
        
        # Add confidence intervals
        # std_rolling_avg = np.std(interpolated_avgs, axis=0)
        # ax.fill_between(np.arange(len(mean_rolling_avg))*120/len(mean_rolling_avg),
        #                mean_rolling_avg - std_rolling_avg,
        #                mean_rolling_avg + std_rolling_avg,
        #                color='gray', alpha=0.2)

    
    plt.tight_layout()
    sns.despine()
    plt.show()

def plot_comparative_wpm(objects, usernames):
    """Plot WPM over time for multiple users with rolling averages and power law fit.
    
    Args:
        objects: List of log data objects
        usernames: List of usernames to plot
    """
    # Create figure with modern styling
    plt.figure(figsize=(12, 8), dpi=100, facecolor='white')
    plt.gca().set_facecolor('white')
    
    # Generate distinct colors for each user
    colors = sns.color_palette("husl", n_colors=len(usernames))
    
    # Store data for analysis
    all_rolling_avgs = []
    all_times = []
    
    # Plot individual user data
    for i, username in enumerate(usernames):
        entries = get_user_entries(objects, username)
        wpm_values, _, _, _, cum_time_elapsed = calculate_metrics(entries)
        
        # Normalize time to 120 minutes
        cum_time_elapsed *= 120 / cum_time_elapsed[-1]
        
        # Plot raw data points
        sns.scatterplot(x=cum_time_elapsed, y=wpm_values, alpha=0.2, 
                       color=colors[i], label=f'{username} (Raw)', 
                       s=50, edgecolor=None)
        
        # Calculate and plot exponential moving average
        rolling_avg = calculate_rolling_average(wpm_values)
        sns.lineplot(x=cum_time_elapsed, y=rolling_avg, alpha=0.8,
                    color=colors[i], label=f'{username} (EMA)',
                    linewidth=2)
        
        all_rolling_avgs.append(rolling_avg)
        all_times.append(cum_time_elapsed)
    
    # Calculate mean trend across users
    min_len = min(len(t) for t in all_times)
    common_time = np.linspace(0, 120, min_len)
    
    interpolated_avgs = []
    for i, ra in enumerate(all_rolling_avgs):
        interp_func = np.interp(common_time, all_times[i], ra)
        interpolated_avgs.append(interp_func)
    
    mean_rolling_avg = np.mean(interpolated_avgs, axis=0)
    
    # Plot group average
    sns.lineplot(x=common_time, y=mean_rolling_avg, color='black',
                linewidth=3, label='Group Average',
                linestyle='-')
    
    # Fit and plot power law curve
    log_x = np.log(common_time + 1)
    log_y = np.log(mean_rolling_avg)
    coeffs = np.polyfit(log_x, log_y, 1)
    a, b = np.exp(coeffs[1]), coeffs[0]
    power_law = a * (common_time + 1)**b
    
    plt.plot(common_time, power_law, color='black',#color='#FF1493',
            linestyle='--', linewidth=2.5,
            label=f'Power Law Fit (y = {a:.2f}x^{b:.2f})')
    
    # Style the plot
    plt.ylabel('Words Per Minute (WPM)', fontsize=14, fontname='Arial', labelpad=12)
    plt.xlabel('Time (minutes)', fontsize=14, fontname='Arial', labelpad=12)
    plt.title('Learning Curve of Four Participants in a Two Hour Pilot Study',
             pad=15, fontsize=16, fontname='Arial', fontweight='medium')
    
    # Customize legend
    plt.legend(frameon=True, facecolor='white', framealpha=0.95,
              edgecolor='lightgray', fontsize=10, ncol=2)
    
    # Add grid and set limits
    plt.grid(True, alpha=0.2, linestyle='-', color='black')
    plt.ylim(0, 10)
    
    # Final styling touches
    sns.despine()
    plt.tight_layout()
    plt.show()

# Load and analyze data
objects = load_data('../../comblog.txt')
get_user_counts(objects)

# Single user analysis
target_username = 'p4'
entries = get_user_entries(objects, target_username)
plot_user_metrics(entries, target_username)

# Multi-user comparative analysis
usernames = ['p1', 'P2', 'P3', 'p4']
plot_comparative_wpm(objects, usernames)

#%%
# Analyze last 10 records for each participant
participants = ['p1', 'P2', 'P3', 'p4']

for participant in participants:
    entries = get_user_entries(objects, participant)[-10:]
    
    # Extract delays from delay_pairs
    delays = []
    for entry in entries:
        if 'delay_pairs' in entry:
            for pair in entry['delay_pairs']:
                if 'delay' in pair:
                    delays.append(pair['delay'])
    
    # Calculate mean and standard deviation
    mean_delay = np.mean(delays) if delays else 0
    std_delay = np.std(delays) if delays else 0
    
    print(f"\nLast 10 entries for {participant}:")
    print(f"Mean delay: {mean_delay:.2f} ms")
    print(f"Std dev: {std_delay:.2f} ms")

#%%
# Analyze last 10 records for each participant
participants = ['p1', 'P2', 'P3', 'p4', 'example', 'eyes2']

# objects = load_data('../../comblog.txt')
objects = load_data('./log.txt')
n_last_entries = 10
for participant in participants:
    entries = get_user_entries(objects, participant)[-n_last_entries:]
    if not entries:
        continue
    
    # Extract delays from delay_pairs
    delays = []
    total_time = 0
    total_gestures = 0
    total_characters = 0
    
    for entry in entries:
        if 'delay_pairs' in entry:
            for pair in entry['delay_pairs']:
                if 'delay' in pair:
                    delays.append(pair['delay'])
        # Add time elapsed
        round_time = round(entry.get('time_elapsed', 0), 2)
        total_time += round_time
        # Add number of gestures (delay_pairs length + 1)
        total_gestures += len(entry.get('delay_pairs', [])) + 1
        round_characters = len(entry.get('target_phrase', '')) - 1
        total_characters += round_characters
        print('round wpm: ', round_characters / round_time * 60 / 5)
    
    # Calculate metrics
    mean_delay = np.mean(delays) if delays else 0
    std_delay = (np.percentile(delays, 90) - np.percentile(delays, 10)) / 2.56 if delays else 0
    avg_gesture_time = total_time / total_gestures if total_gestures > 0 else 0
    avg_wait_time = entries[-1]['delay_pairs'][0]['period'] / 2
    avg_scan_time = avg_gesture_time - avg_wait_time
    avg_wpm = total_characters / total_time * 60 / 5
    
    print(f"\nLast 10 entries for {participant}:")
    print(f"Mean delay: {mean_delay:.3f} s")
    print(f"Std dev: {std_delay:.3f} s")
    print(f"Total time elapsed: {total_time:.3f} s")
    # print(f"Total gestures: {total_gestures}")
    # print(f"Average gesture time: {avg_gesture_time:.3f} s/gesture")
    print(f"Average wait time: {avg_wait_time:.3f} s")
    print(f"Average scan time: {avg_scan_time:.3f} s/gesture")
    print(f"Average WPM: {avg_wpm:.3f} WPM")

#%%
get_user_entries(objects, 'p4')[-1]['delay_pairs'][0]['period']