#%%
import json
data = json.loads(open('blink_log_long.json').read())['history']
data

times = [entry['metadata']['captureTime'] for entry in data]
lefts = [entry['left'] for entry in data]
rights = [entry['right'] for entry in data]
import numpy as np
lefts = np.array(lefts)
rights = np.array(rights)
values = np.maximum(lefts, rights)
times = np.array(times)

from matplotlib import pyplot as plt

plt.vlines(np.arange(40)*1000, 0, 0.5, color='red', linestyle='dashed', linewidth=1)
plt.plot(times, (lefts + rights) / 2, linewidth=1, marker='o', markersize=2)
# plt.figsize(10, 5)
plt.show()
#%%
plt.figure(figsize=(14, 5))
plt.plot(times[0:300], values[0:300], linewidth=1, marker='o', markersize=3)
# blink_times = label(times[200:400], lefts[200:400], rights[200:400], interpolate=True)
# blink_times_no_interpolate = label(times[200:400], lefts[200:400], rights[200:400], interpolate=False)
# plt.vlines(blink_times, 0, 0.5, color='green', linewidth=.4)
# plt.vlines(blink_times_no_interpolate, 0, 0.5, color='blue', linewidth=.4)
plt.show()
#%%
blink_threshold = 0.25
def label(times,lefts,rights, interpolate=False, fancy_interpolate=False, fancy_params=None, c=1):
    assert not (interpolate and fancy_interpolate)
    labels = []
    blinking = False
    cooldown_start = None
    cooldown_time = 250
    blink_times = []
    regimes = []
    ii = []

    values = np.maximum(lefts, rights)

    for i, (time,value) in enumerate(zip(times,values)):
        label = None
        not_cooldown = cooldown_start is None or time - cooldown_start > cooldown_time
        if value > blink_threshold and not blinking and not_cooldown:
            ii.append(i)
            # eyes just closed and we're not in cooldown
            blinking = True
            cooldown_start = time
            # determine the regime with lookahead (asc, straight, cross)
            prev_time, next_time = times[i-1], times[i+1]
            prev_value, next_value = values[i-1], values[i+1]
            slope = (value - prev_value)/(time - prev_time)
            next_slope = (next_value - value)/(next_time - time)
            if prev_value < value and value < next_value:
                regimes.append('asc')
            elif value >= next_value:
                if abs(slope) > c * abs(next_slope):
                    regimes.append('straight')
                else:
                    # print(f"CROSS")
                    # print(f"Time: {time}")
                    # print(f"{prev_value:.2f}, {value:.2f}, {next_value:.2f}")
                    regimes.append('cross')
            else:
                print("BAD REGIME: ", f"{prev_value:.2f}, {value:.2f}, {next_value:.2f}")
            # get the corrected time
            if interpolate:
                interp_frac = (blink_threshold - prev_value) / (value - prev_value)
                # interp_frac **= 0.7
                corrected_time = prev_time + interp_frac * (time - prev_time)
                blink_times.append(corrected_time)
            elif fancy_interpolate:
                assert fancy_params is not None
                fp = fancy_params
                # prior_h0  -- mixing constant for straight
                straight_ll = -np.log(fp['ss_sigma']) - ((slope - fp['ss_mean']) / fp['ss_sigma'])**2 / 2
                cross_ll = -np.log(fp['cross_sigma']) - ((slope - fp['cross_mean']) / fp['cross_sigma'])**2 / 2
                straight_logpost_Z = np.log(fp['prior_h0']) + straight_ll
                cross_logpost_Z = np.log(1 - fp['prior_h0']) + cross_ll
                straight_post = np.exp(straight_logpost_Z) / (np.exp(straight_logpost_Z) + np.exp(cross_logpost_Z))
                cross_post = 1 - straight_post
                # straight_post, cross_post = 1, 0  # equivalent to standard interpolate=True
                # straight_post, cross_post = 0, 1  # assumes a constant speed
                # If straight do standard interpolation
                straight_interp_frac = (blink_threshold - prev_value) / (value - prev_value)
                # If cross, don't try to infer Altitude, just use ss_mean
                dummy_value = prev_value + fp['ss_mean']*(time - prev_time)  # what would we be at if we hadn't crossed?
                cross_interp_frac = (blink_threshold - prev_value) / (dummy_value - prev_value)
                # take weighted average of the interp fracs
                interp_frac = straight_post * straight_interp_frac + cross_post * cross_interp_frac
                # interp_frac **= 0.7
                corrected_time = prev_time + interp_frac * (time - prev_time)
                blink_times.append(corrected_time)
            else:
                blink_times.append(time)
            #
            label = 1
        elif value<blink_threshold and blinking:
            blinking = False
            label = 0
        else:
            label = 0
        assert label is not None
        labels.append(label)

    # print("Regimes: ", regimes)
    bincounts = np.bincount([{"straight": 0, "cross": 1, "asc": 2}[x] for x in regimes])
    # print(bincounts)
    return blink_times, bincounts, ii, regimes

label(times, lefts, rights, interpolate=True, c=2)
c_space = np.linspace(1, 5, 100)
bincounts = [label(times, lefts, rights, interpolate=True, c=c)[1] for c in c_space]
straight_percs = [(x[0])/(x[0]+x[1]+x[2]) for x in bincounts]
# plt.plot(c_space, straight_percs)
# plt.plot(c_space, 1/(c_space+1))
# plt.plot(c_space, np.log(1/(c_space+1))*[x[0] for x in bincounts] + np.log(c_space/(c_space + 1))*[x[1] for x in bincounts])# - (c_space-2)**2)

C = 2.5  # TODO: choose C by what gives the most informative split on slope
blink_times, bincounts, ii, regimes = label(times, lefts, rights, interpolate=True, c=C)
list(zip(ii, regimes))
# straight_labels = [1 if regime == 'asc' or (values[i] - values[i-1])/(times[i] - times[i-1]) > 0.009 else 0 for i, regime in zip(ii, regimes)]
# ss = [(values[i] - values[i-1])/(times[i] - times[i-1]) for i, label in zip(ii, straight_labels) if label == 1]
# crosses = [(values[i] - values[i-1])/(times[i] - times[i-1]) for i, label in zip(ii, straight_labels) if label == 0]
ss = [(values[i] - values[i-1])/(times[i] - times[i-1]) for i,regime in zip(ii, regimes) if regime != 'cross']
crosses = [(values[i] - values[i-1])/(times[i] - times[i-1]) for i,regime in zip(ii, regimes) if regime == 'cross']

plt.hist(ss, bins=100)
plt.hist(crosses, bins=100)
np.std(ss), np.std(crosses)
fancy_params = {
    'prior_h0': len(ss)/(len(ss) + len(crosses)),
    'ss_mean': np.mean(ss),
    'ss_sigma': np.std(ss),
    'cross_mean': np.mean(crosses),
    'cross_sigma': np.std(crosses),
}
super_delays = (np.array(label(times, lefts, rights, fancy_interpolate=True, fancy_params=fancy_params)[0]) + 500) % 1000 - 500
np.std(super_delays), fancy_params['prior_h0']

#%%
good_delays = (np.array(label(times, lefts, rights, interpolate=True)[0]) + 500) % 1000 - 500
# good_delays = (np.array(label(times, lefts, rights, fancy_interpolate=True, fancy_params=fancy_params)[0]) + 500) % 1000 - 500
bad_delays = (np.array(label(times, lefts, rights, interpolate=False)[0]) + 500) % 1000 - 500
deltas = good_delays - bad_delays
good_delays -= np.mean(good_delays)
bad_delays -= np.mean(bad_delays)


good_std = np.std(good_delays)
bad_std = np.std(bad_delays)

plt.plot(bad_delays, deltas, 'o', markersize=1, label=f"Correction from interpolation (std = {bad_std:.4f})")
plt.vlines(good_delays, 0, 3, color='green', linewidth=.6, label=f"delays with interpolation (std = {good_std:.4f})")
plt.vlines(bad_delays, 2, 5, color='blue', linewidth=.6, label="raw delays")

# Fit and plot line of best fit
coeffs = np.polyfit(bad_delays, deltas, 1)
fit_line = np.polyval(coeffs, bad_delays)
# Calculate the correct r^2 value
ss_res = np.sum((deltas - fit_line) ** 2)
ss_tot = np.sum((deltas - np.mean(deltas)) ** 2)
r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
plt.plot(bad_delays, fit_line, color='red', label=f'Best fit (slope = {coeffs[0]:.2f}; r² = {r_squared:.2f})')

plt.legend(loc='lower left')
plt.show()

#%%
len(good_delays)
# yn1s = np.array([values[i-3] for i in ii])
y0s = np.array([values[i-2] for i in ii])
y1s = np.array([values[i-1] for i in ii])
y2s = np.array([values[i] for i in ii])
dts = np.array([times[i] - times[i-1] for i in ii])
# dts = (dts - np.mean(dts))/np.std(dts)
len(y1s)
# y1, y2, and y3 are useless for correction, but y0 is useful, and to a lesser extent dts as well.
# X = np.column_stack([np.ones_like(y1s), (y0s - np.mean(y0s))/np.std(y0s), dts])#, y1s*y2s, y1s**2, y2s**2])
X = np.column_stack([np.ones_like(y1s), dts])#, y1s*y2s, y1s**2, y2s**2])
CUT = 100
coeffs, residuals, rank, s = np.linalg.lstsq(X[:CUT], good_delays[:CUT], rcond=None)
delays_predicted = X @ coeffs
print("std(good_delays[CUT:]) = ", np.std(good_delays[CUT:]))
print("std(good_delays[:CUT]) = ", np.std(good_delays[:CUT]))
print("std(good_delays[CUT:] - delays_predicted[CUT:]) = ", np.std(good_delays[CUT:] - delays_predicted[CUT:]))
print("std(good_delays[:CUT] - delays_predicted[:CUT]) = ", np.std(good_delays[:CUT] - delays_predicted[:CUT]))
plt.hist(y0s, bins=100, alpha=0.3)
plt.hist(y1s, bins=100, alpha=0.3)
plt.hist(y2s, bins=100, alpha=0.3)
coeffs
#%%
plt.hist(dts, bins=100)