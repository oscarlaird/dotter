
#%%
import nest_asyncio
nest_asyncio.apply()
import torch
torch.cuda.is_available()

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import asyncio
import time
import numpy as np

# Global variables for model and tokenizer
model_name = "gpt2-xl"
tokenizer = None
model = None
cache = None

def initialize_model_and_tokenizer():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.cuda()
initialize_model_and_tokenizer()

#%%
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Read and process log file
delays = []
with open('log.txt', 'r') as f:
    for line in f:
        try:
            data = json.loads(line)
            if data.get('username') == 'oscar':
                delay_pairs = data.get('delay_pairs', [])
                delays.extend(pair['delay'] for pair in delay_pairs)
        except json.JSONDecodeError:
            continue

# Create histogram with smaller bins and linear x-axis
plt.figure(figsize=(10, 6))
counts, bins, _ = plt.hist(delays, bins=200, density=True, alpha=0.7, log=True)
plt.xlabel('Delay (seconds)')
plt.ylabel('Log Density')
plt.xlim(-.2, .2)
plt.title('Distribution of Delays for User "oscar" (Log Scale)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
