#%%
import torch
import torch.nn as nn
import torch.optim as optim

import core

dummy_data = \
[
    # session0
    # high delay, consistent
    [
        [0.5],
        # string0
        [ 0.7, 0.6, 0.6, 0.6]*5,
        # string1
        [ 0.8, 0.7, 0.7, 0.7]*5,
    ],
    # session1
    # less delay, less consitent
    [
        [0.5],
        # string0
        [ 0.2, 0.3, 0.4]*5,
        # string1
        [ 0.3, 0.3, 0.6]*5,
        # string2
        # really wild
        [ -0.1, -0.1, 0.6, 0.6]*5,
    ],
]

assert torch.cuda.is_available()
device = torch.device("cuda")

# ij0; string mean
# 100ms
        # i00; typical string mean for session
        # 100ms
                # 000; typical session mean
                # 100ms
                # 001; variation between sessions wrt session mean
                # 25ms
        # i01; string mean variation within session
        # 10ms
                # 010; typical session control over mean
                # 10ms
                # 011; variation between sessions wrt session control over mean
                # 1% (ignore)
# ij1; string dev
# 40ms
        # i10; typical string dev for session
        # 40ms
                # 100; typical dev
                # 40ms
                # 101; how much do sessions differ in dev?
                # 30%
        # i11; how much does session control string dev?
        # 10ms
                # 110; typical session control over dev
                # 10ms
                # 111; variation between sessions wrt session control over dev
                # 1% (ignore)

# initialize all means and log deviations to 0
priors = torch.tensor([0.0]*8, device=device)
model = core.VNode(dummy_data, priors)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=4e-2)

loss_history = []

for i in range(2_000):
    optimizer.zero_grad()
    loss = -core.elbo(model)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if i % 50 == 0:
        avg_loss = sum(loss_history[-50:]) / 50
        print(f"Avg Loss after {i} iterations: {avg_loss}")
   
#%%
# inspect
print(model.down[0].down[0].down[0].val)
print("SESSION 1")
print(model.down[0].down[0].val)
print(model.down[0].down[1].val)
print(model.down[0].down[2].val)
print("SESSION 2")
print(model.down[1].down[0].val)
print(model.down[1].down[1].val)
print(model.down[1].down[2].val)
print(model.down[1].down[3].val)
print(model.val)