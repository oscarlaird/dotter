import numpy as np

max_x = 2.0
RES = 11  # number of grid points
delta = max_x / (RES - 1)



joint = np.zeros((RES, RES), dtype=float)
joint += 1
# normalize
joint = joint / joint.sum()
# N.B: these are probabilities, not densities

# compute marginals
margX = joint.sum(axis=0)
margY = joint.sum(axis=1)

margZ = np.zeros(2 * RES - 1)
for i in range(RES):
    margZ[i:RES+i] += joint[i]


# entropies
# H(X) := E[ -log P(x) ]
# Since we want the differential entropy we want to use the density which is point_prob / delta
def entropy(distrib):
    prob_density = distrib / delta
    info = -np.log(prob_density)
    expected_info = (distrib * info).sum()
    return expected_info

def cond_entropy(joint):
    # compute H(Y|X)  # X is along axis=1, Y is along axis=0
    margX = joint.sum(axis=0)
    entropies = [entropy(joint[:,i] / margX[i]) for i in range(len(joint[0]))]
    expected_entropy = (margX * np.array(entropies)).sum()
    return expected_entropy

def EV(distrib):
    x_vals = np.arange(RES) * delta
    return (x_vals * distrib).sum()

HX = entropy(margX)
HY = entropy(margY)
HZ = entropy(margZ)
HYX = cond_entropy(joint)

EX = EV(margX)
EY = EV(margY)
EZ = EX + EY

objective = (HZ - HYX) / EZ

