#%%
# log is lines of json like
# {
#     "best_string": "hello",
#     "target_phrase": "hello",
#     "timer_fracs": [0.1, 0.2, 0.3],
#     "wpm_timer_start": 1000,
#     "time_elapsed": 1000,
#     "wpm": 20
# }
#%%
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Read and parse log file
with open('log.txt', 'r') as f:
    logs = [json.loads(line) for line in f][-1:]

print(f"Analyzed {len(logs)} typing sessions\n")

# Collect all timer fractions across sessions
all_timer_fracs = []

for i, log in enumerate(logs):
    print(f"Session {i+1}:")
    print(f"Target phrase: {log['target_phrase']}")
    print(f"Actual typed: {log['best_string']}")
    print(f"WPM: {log['wpm']:.1f}")
    print(f"Time elapsed: {log['time_elapsed']:.1f}s")
    print(f"Number of gestures: {len(log['timer_fracs'])}")
    print("---")
    
    all_timer_fracs.extend(log['timer_fracs'])

# Calculate statistics
mu = np.mean(all_timer_fracs)
sigma = np.std(all_timer_fracs)

print(f"\nOverall statistics:")
print(f"Total number of gestures: {len(all_timer_fracs)}")
print(f"Mean timing fraction: {mu:.3f}")
print(f"Standard deviation: {sigma:.3f}")

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(all_timer_fracs, bins=50, density=True, alpha=0.7, color='blue')

# Add fitted normal distribution curve
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, sigma)
plt.plot(x, p, 'k', linewidth=2)

plt.title('Distribution of Timer Fractions')
plt.xlabel('Timer Fraction')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

# Add text box with statistics
plt.text(0.95, 0.95, f'μ = {mu:.3f}\nσ = {sigma:.3f}', 
         transform=plt.gca().transAxes,
         verticalalignment='top',
         horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.show()
