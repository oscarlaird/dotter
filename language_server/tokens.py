from transformers import AutoTokenizer
import random
import string
import math

# Initialize the TinyLlama tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Get all tokens from the vocabulary
vocab = tokenizer.get_vocab()
all_tokens = list(vocab.keys())

# Get clean tokens (only lowercase alpha and space, excluding special tokens)
clean_tokens = [
    t for t in vocab.keys() 
    if t not in tokenizer.all_special_tokens 
    and all(c in string.ascii_lowercase or c == '▁' for c in t)
]

# Divide clean tokens into those with leading space and those without
leading_space = [t for t in clean_tokens if t.startswith('▁')]
no_leading_space = [t for t in clean_tokens if not t.startswith('▁')]

# Find prefixes to filter out from no_leading_space
sorted_clean_tokens = sorted(clean_tokens)
clean_token_prefixes = set()
for i in range(len(sorted_clean_tokens) - 1):
    if sorted_clean_tokens[i+1].startswith(sorted_clean_tokens[i]):
        clean_token_prefixes.add(sorted_clean_tokens[i])

no_leading_space_no_prefix = [t for t in no_leading_space if t not in clean_token_prefixes]

print(f"Total clean tokens: {len(clean_tokens)}")
print(f"Clean tokens with leading space: {len(leading_space)}")
print(f"Clean tokens without leading space: {len(no_leading_space)}")
print(f"Clean tokens without leading space and not a prefix: {len(no_leading_space_no_prefix)}")
print()

# Print 10 random tokens
print("10 Random Tokens:")
print("=" * 50)
random_tokens = random.sample(all_tokens, min(10, len(all_tokens)))
for i, token in enumerate(random_tokens, 1):
    token_id = vocab[token]
    print(f"{i}. Token: {repr(token)} (ID: {token_id})")

print("\n" + "=" * 50)
print("\n10 Random Merge Rules:")
print("=" * 50)

# Try to get merge rules from the tokenizer
merges = None
try:
    # For tokenizers library BPE models, merges are stored in the model's vocabulary
    # We need to access them through the backend_tokenizer
    if hasattr(tokenizer, 'backend_tokenizer'):
        backend = tokenizer.backend_tokenizer
        if hasattr(backend, 'model'):
            model = backend.model
            # BPE models in tokenizers library store merges differently
            # Try to get merges from the model's internal state
            if hasattr(model, 'merges'):
                merges = list(model.merges)
            elif hasattr(model, '_merges'):
                merges = list(model._merges)
            elif hasattr(model, 'get_merges'):
                merges = model.get_merges()
            else:
                # Try to reconstruct merges from the vocabulary
                # BPE merges are typically stored as tuples in the model
                # Let's try accessing through the model's internal structure
                try:
                    # For tokenizers BPE, merges might be in the model's state
                    import tokenizers
                    if isinstance(model, tokenizers.models.BPE):
                        # Try to get merges from the model's saved state
                        # The merges are usually stored as a list of tuples or strings
                        if hasattr(model, 'merges'):
                            merges = model.merges
                        # Alternative: try to access through __dict__ or _merges
                        elif '_merges' in model.__dict__:
                            merges = model.__dict__['_merges']
                        # Another approach: get merges from the tokenizer's saved files
                        # But first, let's try to serialize and see what we get
                        else:
                            # Last resort: try to load from tokenizer.json if we can find it
                            # But for now, let's try one more thing - check if there's a way to get merges
                            # through the tokenizer's save method or through the model's internal state
                            pass
                except Exception as e2:
                    pass
    
    # If we still don't have merges, try loading from tokenizer.json directly
    if merges is None:
        try:
            import json
            import os
            from pathlib import Path
            
            # Try to find the tokenizer.json file in the cache
            # The tokenizer should have a path to its files
            if hasattr(tokenizer, 'name_or_path'):
                # Try to load from HuggingFace cache
                cache_path = os.path.expanduser("~/.cache/huggingface/hub")
                model_name_safe = model_name.replace("/", "--")
                # Look for tokenizer.json in the cache
                possible_paths = [
                    os.path.join(cache_path, f"models--{model_name_safe}", "snapshots", "*", "tokenizer.json"),
                    os.path.join(cache_path, f"models--{model_name_safe}", "*", "tokenizer.json"),
                ]
                
                # Also try the tokenizer's actual files if they exist
                if hasattr(tokenizer, 'tokenizer_file') and os.path.exists(tokenizer.tokenizer_file):
                    with open(tokenizer.tokenizer_file, 'r') as f:
                        tokenizer_data = json.load(f)
                        if 'model' in tokenizer_data and 'merges' in tokenizer_data['model']:
                            merges = tokenizer_data['model']['merges']
                
                # Try to find tokenizer.json in cache directories
                if merges is None:
                    import glob
                    for pattern in possible_paths:
                        matches = glob.glob(pattern)
                        if matches:
                            with open(matches[0], 'r') as f:
                                tokenizer_data = json.load(f)
                                if 'model' in tokenizer_data and 'merges' in tokenizer_data['model']:
                                    merges = tokenizer_data['model']['merges']
                                    break
        except Exception as e3:
            pass
    
    if merges:
        random_merges = random.sample(merges, min(10, len(merges)))
        for i, merge in enumerate(random_merges, 1):
            print(f"{i}. Merge: {merge}")
    else:
        print("Could not find merges in tokenizer.")
        print("Note: BPE merges may not be directly accessible through the tokenizer API.")
            
except Exception as e:
    import traceback
    print(f"Error accessing merge rules: {e}")
    traceback.print_exc()

# Sanity checks
print("\n" + "=" * 50)
print("Sanity Checks:")
print("=" * 50)

# Sanity check 1: Verify we can decode tokens back to their original form
print("\n1. Testing token decode/encode roundtrip:")
test_tokens = random.sample(clean_tokens, min(10, len(clean_tokens)))
for token in test_tokens:
    token_id = vocab[token]
    # Decode token to get its string representation
    decoded_str = tokenizer.decode([token_id])
    # Encode the decoded string, prepending '-' to prevent implicit space injection
    encoded = tokenizer.encode('-' + decoded_str, add_special_tokens=False)
    if encoded and encoded[0] == 448:  # 448 is the ID for '-'
        encoded = encoded[1:]
    # Check if it matches the original token ID
    matches = encoded == [token_id]
    print(f"   Token: {repr(token)} (ID: {token_id}) -> Decoded: {repr(decoded_str)} -> Matches: {matches}")

# Sanity check 2: Test a few concatenations manually
print("\n2. Testing a few concatenations manually:")
for _ in range(5):
    token1 = random.choice(leading_space)
    token2 = random.choice(no_leading_space)
    token1_id = vocab[token1]
    token2_id = vocab[token2]
    # Decode tokens to get their actual string representation
    token1_str = tokenizer.decode([token1_id])
    token2_str = tokenizer.decode([token2_id])
    concatenated = token1_str + token2_str
    
    is_leading = token1.startswith('▁')
    if is_leading:
        # For leading space tokens, allow the implicit space injection
        encoded = tokenizer.encode(concatenated, add_special_tokens=False)
    else:
        # For no leading space tokens, prevent the implicit space injection
        encoded = tokenizer.encode('-' + concatenated, add_special_tokens=False)
        if encoded and encoded[0] == 448:
            encoded = encoded[1:]
            
    expected = [token1_id, token2_id]
    matches = encoded == expected
    print(f"   {repr(token1)} (ID: {token1_id}) + {repr(token2)} (ID: {token2_id})")
    print(f"     Decoded: {repr(token1_str)} + {repr(token2_str)} = {repr(concatenated)}")
    print(f"     Encoded: {encoded}, Expected: {expected}, Matches: {matches}")
    if not matches:
        print(f"     Decoded tokens: {[tokenizer.decode([t]) for t in encoded]}")

# Main experiment: larger sample
NUM_FIRST_TOKENS = 1000
NUM_SECOND_TOKENS_PER_FIRST = 100
TOTAL_TRIALS = NUM_FIRST_TOKENS * NUM_SECOND_TOKENS_PER_FIRST

print("\n" + "=" * 50)
print("Main Experiment: Token Concatenation Test")
print("=" * 50)
print(f"Picking {NUM_FIRST_TOKENS} random clean tokens (using random.sample)...")
print(f"For each, concatenating with {NUM_SECOND_TOKENS_PER_FIRST} different random no_leading_space_no_prefix tokens...")
print(f"Total trials: {TOTAL_TRIALS:,}")
print()

# Pick random clean tokens (using random.sample for true random sampling)
sample_clean_tokens = random.sample(clean_tokens, min(NUM_FIRST_TOKENS, len(clean_tokens)))

print(f"Selected {len(sample_clean_tokens)} random clean tokens (using random.sample)")
print(f"  - {sum(1 for t in sample_clean_tokens if t.startswith('▁'))} with leading space")
print(f"  - {sum(1 for t in sample_clean_tokens if not t.startswith('▁'))} without leading space")
print(f"For each, we'll test with {NUM_SECOND_TOKENS_PER_FIRST} random no_leading_space_no_prefix tokens (using random.sample)")
print(f"Total trials: {len(sample_clean_tokens)} × {NUM_SECOND_TOKENS_PER_FIRST} = {len(sample_clean_tokens) * NUM_SECOND_TOKENS_PER_FIRST:,}")
print()

matches_count = 0
total_trials = 0
match_examples = []
non_match_examples = []
# Track pass rate for each first token
first_token_pass_rates = []
# Track pass rate for each second token
second_token_pass_rates = {}
second_token_trials = {}

for i, token1 in enumerate(sample_clean_tokens):
    if (i + 1) % 100 == 0:
        print(f"Processing token {i+1}/{len(sample_clean_tokens)}...")
    
    # Get token1 ID and decode to string
    token1_id = vocab[token1]
    token1_str = tokenizer.decode([token1_id])
    
    # Pick random no_leading_space_no_prefix tokens (using random.sample for true random sampling)
    sample_no_leading = random.sample(no_leading_space_no_prefix, min(NUM_SECOND_TOKENS_PER_FIRST, len(no_leading_space_no_prefix)))
    
    # Track matches for this first token
    token1_matches = 0
    token1_trials = 0
    
    is_leading = token1.startswith('▁')
    
    for token2 in sample_no_leading:
        # Get token2 ID and decode to string
        token2_id = vocab[token2]
        token2_str = tokenizer.decode([token2_id])
        
        # Concatenate the decoded strings
        concatenated = token1_str + token2_str
        
        if is_leading:
            # Tokenize normally
            encoded = tokenizer.encode(concatenated, add_special_tokens=False)
        else:
            # Tokenize with '-' prepended to bypass implicit start-of-string space injection
            encoded = tokenizer.encode('-' + concatenated, add_special_tokens=False)
            if encoded and encoded[0] == 448:
                encoded = encoded[1:]
        
        # Track trials for this second token
        if token2 not in second_token_pass_rates:
            second_token_pass_rates[token2] = 0
            second_token_trials[token2] = 0
        second_token_trials[token2] += 1
        
        # Check if it matches exactly [token1_id, token2_id]
        if encoded == [token1_id, token2_id]:
            matches_count += 1
            token1_matches += 1
            # Track matches for this second token
            second_token_pass_rates[token2] += 1
            # Collect match examples (up to 10)
            if len(match_examples) < 10:
                match_examples.append({
                    'token1': token1,
                    'token2': token2,
                    'token1_id': token1_id,
                    'token2_id': token2_id,
                    'token1_str': token1_str,
                    'token2_str': token2_str,
                    'concatenated': concatenated,
                    'encoded': encoded
                })
        else:
            # Collect non-match examples (up to 10)
            if len(non_match_examples) < 10:
                non_match_examples.append({
                    'token1': token1,
                    'token2': token2,
                    'token1_id': token1_id,
                    'token2_id': token2_id,
                    'token1_str': token1_str,
                    'token2_str': token2_str,
                    'concatenated': concatenated,
                    'encoded': encoded,
                    'expected': [token1_id, token2_id]
                })
        
        token1_trials += 1
        total_trials += 1
    
    # Calculate pass rate for this first token
    pass_rate = (token1_matches / token1_trials * 100) if token1_trials > 0 else 0
    first_token_pass_rates.append(pass_rate)

print("\n" + "=" * 50)
print("Results:")
print("=" * 50)
print(f"Total trials: {total_trials}")
print(f"Trials where concatenation tokenizes as the two original tokens: {matches_count}")
print(f"Percentage: {matches_count/total_trials*100:.2f}%")
print(f"Trials where concatenation tokenizes differently: {total_trials - matches_count}")
print(f"Percentage: {(total_trials - matches_count)/total_trials*100:.2f}%")

# Collect histogram data: number of tokens produced when concatenating
# Note: We'll use a sample for this to avoid re-running all trials
token_count_distribution = {}
sample_for_hist = random.sample(sample_clean_tokens, min(100, len(sample_clean_tokens)))
for i, token1 in enumerate(sample_for_hist):
    token1_id = vocab[token1]
    token1_str = tokenizer.decode([token1_id])
    sample_no_leading = random.sample(no_leading_space_no_prefix, min(100, len(no_leading_space_no_prefix)))
    
    is_leading = token1.startswith('▁')
    
    for token2 in sample_no_leading:
        token2_id = vocab[token2]
        token2_str = tokenizer.decode([token2_id])
        concatenated = token1_str + token2_str
        
        if is_leading:
            encoded = tokenizer.encode(concatenated, add_special_tokens=False)
        else:
            # Tokenize with '-' prepended to bypass implicit start-of-string space injection
            encoded = tokenizer.encode('-' + concatenated, add_special_tokens=False)
            if encoded and encoded[0] == 448:
                encoded = encoded[1:]
            
        num_tokens = len(encoded)
        token_count_distribution[num_tokens] = token_count_distribution.get(num_tokens, 0) + 1

# Display histogram of token counts
print("\n" + "=" * 50)
print("Histogram: Number of tokens produced when concatenating two tokens")
print("=" * 50)
max_count = max(token_count_distribution.values())
max_tokens = max(token_count_distribution.keys())
bar_width = 60

for num_tokens in sorted(token_count_distribution.keys()):
    count = token_count_distribution[num_tokens]
    bar_length = int((count / max_count) * bar_width)
    bar = '█' * bar_length
    percentage = (count / total_trials) * 100
    print(f"{num_tokens:2d} tokens: {bar} {count:5d} ({percentage:5.2f}%)")

# Display histogram of pass rates by first token
print("\n" + "=" * 50)
print("Histogram: Pass rate distribution for first tokens (10% buckets)")
print("=" * 50)

# Create buckets: 0-10%, 10-20%, ..., 90-100%
buckets = {}
for i in range(10):
    bucket_min = i * 10
    bucket_max = (i + 1) * 10
    bucket_key = f"{bucket_min}-{bucket_max}%"
    buckets[bucket_key] = 0

# Also handle exactly 100% separately, or include in 90-100%
for pass_rate in first_token_pass_rates:
    if pass_rate == 100:
        buckets["90-100%"] += 1
    else:
        bucket_idx = int(pass_rate // 10)
        bucket_key = f"{bucket_idx * 10}-{(bucket_idx + 1) * 10}%"
        buckets[bucket_key] += 1

# Display histogram
max_bucket_count = max(buckets.values())
bar_width = 60

for bucket_key in sorted(buckets.keys(), key=lambda x: int(x.split('-')[0])):
    count = buckets[bucket_key]
    bar_length = int((count / max_bucket_count) * bar_width) if max_bucket_count > 0 else 0
    bar = '█' * bar_length
    percentage = (count / len(first_token_pass_rates)) * 100
    print(f"{bucket_key:12s}: {bar} {count:3d} first tokens ({percentage:5.2f}%)")

# Calculate binary entropy for each first token
print("\n" + "=" * 50)
print("Monte Carlo Estimation: Expected Information (Entropy) of a Random First Token")
print("=" * 50)

def binary_entropy_bits(p):
    """Calculate binary entropy in bits for probability p"""
    if p == 0 or p == 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

# Calculate entropy for each first token based on its pass rate
first_token_entropies = []
for pass_rate_pct in first_token_pass_rates:
    pass_rate_prob = pass_rate_pct / 100.0
    entropy = binary_entropy_bits(pass_rate_prob)
    first_token_entropies.append(entropy)

# Monte Carlo estimation: sample random first tokens with replacement and calculate expected entropy
NUM_MONTE_CARLO_SAMPLES = 10000
monte_carlo_entropies = random.choices(first_token_entropies, k=NUM_MONTE_CARLO_SAMPLES)
expected_entropy = sum(monte_carlo_entropies) / len(monte_carlo_entropies)

print(f"Number of first tokens analyzed: {len(first_token_pass_rates)}")
print(f"Monte Carlo samples: {len(monte_carlo_entropies):,}")
print(f"Expected information (entropy) of a random first token: {expected_entropy:.4f} bits")
print(f"\nNote: Binary entropy ranges from 0 bits (certainty) to 1 bit (maximum uncertainty)")
print(f"      A pass rate of 50% gives maximum entropy of 1 bit")
print(f"      Lower or higher pass rates give lower entropy")

# Calculate pass rates for second tokens
second_token_pass_rate_list = []
for token2, matches in second_token_pass_rates.items():
    trials = second_token_trials[token2]
    if trials > 0:
        pass_rate = (matches / trials * 100)
        second_token_pass_rate_list.append(pass_rate)

# Display histogram of pass rates by second token
print("\n" + "=" * 50)
print("Histogram: Pass rate distribution for second tokens (10% buckets)")
print("=" * 50)

# Create buckets: 0-10%, 10-20%, ..., 90-100%
second_buckets = {}
for i in range(10):
    bucket_min = i * 10
    bucket_max = (i + 1) * 10
    bucket_key = f"{bucket_min}-{bucket_max}%"
    second_buckets[bucket_key] = 0

# Also handle exactly 100% separately, or include in 90-100%
for pass_rate in second_token_pass_rate_list:
    if pass_rate == 100:
        second_buckets["90-100%"] += 1
    else:
        bucket_idx = int(pass_rate // 10)
        bucket_key = f"{bucket_idx * 10}-{(bucket_idx + 1) * 10}%"
        second_buckets[bucket_key] += 1

# Display histogram
max_second_bucket_count = max(second_buckets.values()) if second_buckets.values() else 1
bar_width = 60

for bucket_key in sorted(second_buckets.keys(), key=lambda x: int(x.split('-')[0])):
    count = second_buckets[bucket_key]
    bar_length = int((count / max_second_bucket_count) * bar_width) if max_second_bucket_count > 0 else 0
    bar = '█' * bar_length
    percentage = (count / len(second_token_pass_rate_list)) * 100 if second_token_pass_rate_list else 0
    print(f"{bucket_key:12s}: {bar} {count:3d} second tokens ({percentage:5.2f}%)")

# Display examples
print("\n" + "=" * 50)
print("10 Examples Where It Matches:")
print("=" * 50)
for i, ex in enumerate(match_examples, 1):
    print(f"\n{i}. Token1: {repr(ex['token1'])} (ID: {ex['token1_id']})")
    print(f"   Token2: {repr(ex['token2'])} (ID: {ex['token2_id']})")
    print(f"   Decoded: {repr(ex['token1_str'])} + {repr(ex['token2_str'])} = {repr(ex['concatenated'])}")
    print(f"   Encoded: {ex['encoded']} ✓")

print("\n" + "=" * 50)
print("10 Examples Where It Doesn't Match:")
print("=" * 50)
for i, ex in enumerate(non_match_examples, 1):
    print(f"\n{i}. Token1: {repr(ex['token1'])} (ID: {ex['token1_id']})")
    print(f"   Token2: {repr(ex['token2'])} (ID: {ex['token2_id']})")
    print(f"   Decoded: {repr(ex['token1_str'])} + {repr(ex['token2_str'])} = {repr(ex['concatenated'])}")
    print(f"   Expected: {ex['expected']}")
    print(f"   Encoded: {ex['encoded']} ✗")
    decoded_tokens = [tokenizer.decode([t]) for t in ex['encoded']]
    print(f"   Decoded tokens: {decoded_tokens}")
