#%%
# !pip install -r requirements.txt
import nest_asyncio
nest_asyncio.apply()
import torch
torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "mps"

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import asyncio
import time
import numpy as np

# Global variables for model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # A small Llama model from Hugging Face
# model_name = "gpt2-xl"  # A small Llama model from Hugging Face
tokenizer = None
model = None
cache = None

def initialize_model_and_tokenizer():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # model.cuda()
    model.to(device)
initialize_model_and_tokenizer()

#%%
# tokenizer.vocab["<|endoftext|>"]
# newline_token_llama = 13
# tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.encode('\n\n\n')

# #%% evaluate the model
# phrases_file = "./src/lib/phrases.txt"
# with open(phrases_file, 'r') as f:
#     phrases = f.read().splitlines()
# # prepend_tokens = [tokenizer.bos_token_id] if model_name == "gpt2-xl" else []
# prompt = """my watch fell in the water
# prevailing wind from the east
# never too rich and never too thin
# breathing is difficult
# i can see the rings on saturn
# """
# tokenizer.encode(prompt + "i am okay")

# #%%
# phrase_tokens = [tokenizer.encode(phrase) for phrase in phrases]
# print(phrase_tokens[7])
# import torch.nn.functional as F
# def bits_in_phrase(prompt, phrase):
#     prompt_len = len(tokenizer.encode(prompt))
#     phrase_len = len(tokenizer.encode(prompt + phrase)) - prompt_len
#     with torch.no_grad():
#         tokens = tokenizer.encode(prompt + phrase)
#         input_ids = torch.tensor(tokens).unsqueeze(0).cuda()
#         outputs = model(input_ids, return_dict=True)
#     logits = outputs.logits[0]  # Take first batch item
#     shift_logits = logits[prompt_len-1:-1, :].contiguous()  # Remove batch dimension since we took first item
#     shift_labels = input_ids[0, prompt_len:].contiguous()  # Also take first batch item and shift
#     loss = F.cross_entropy(
#         shift_logits.view(-1, shift_logits.size(-1)),
#         shift_labels.view(-1)
#     )
#     return loss * phrase_len
# phrase_no = 7
# print(phrases[phrase_no])
# bits_in_phrase(prompt='', phrase=phrases[phrase_no])
# # from tqdm import tqdm
# # phrase_tokens = phrase_tokens[:10]
# # bpts = [bits_in_phrase(tokens, include_eos=True) for tokens in tqdm(phrase_tokens)]
# # total_bits = sum((bpts[i] * len(phrase_tokens[i])) for i in range(len(phrase_tokens)))
# # total_bits / (sum(len(phrase) for phrase in phrases))
# # phrase_tokens[1]
# # bpts[1]
# from tqdm import tqdm
# bit_lengths = {}
# for phrase in tqdm(phrases):
#     bit_lengths[phrase] = {
#         'bits_no_prompt_no_newline': bits_in_phrase(prompt='', phrase=phrase),
#         'bits_no_prompt_with_newline': bits_in_phrase(prompt='', phrase=phrase + '\n'),
#         'bits_with_prompt_no_newline': bits_in_phrase(prompt=prompt, phrase=phrase),
#         'bits_with_prompt_with_newline': bits_in_phrase(prompt=prompt, phrase=phrase + '\n'),
#     }
# #%%
# bit_lengths
# total_bits_with_prompt_no_newline = sum(bit_lengths[phrase]['bits_no_prompt_no_newline'] for phrase in phrases)
# total_bits_with_prompt_with_newline = sum(bit_lengths[phrase]['bits_no_prompt_with_newline'] for phrase in phrases)
# newline_overhead = total_bits_with_prompt_with_newline / total_bits_with_prompt_no_newline
# newline_overhead

# #%%
# total_bits_no_prompt_no_newline = sum(bit_lengths[phrase]['bits_no_prompt_no_newline'] for phrase in phrases)
# total_bits_no_prompt_with_newline = sum(bit_lengths[phrase]['bits_no_prompt_with_newline'] for phrase in phrases)
# total_bits_with_prompt_no_newline = sum(bit_lengths[phrase]['bits_with_prompt_no_newline'] for phrase in phrases)
# total_bits_with_prompt_with_newline = sum(bit_lengths[phrase]['bits_with_prompt_with_newline'] for phrase in phrases)

# print(f"Total bits without prompt, without newline: {total_bits_no_prompt_no_newline:.2f}")
# print(f"Total bits without prompt, with newline: {total_bits_no_prompt_with_newline:.2f}")
# print(f"Total bits with prompt, without newline: {total_bits_with_prompt_no_newline:.2f}")
# print(f"Total bits with prompt, with newline: {total_bits_with_prompt_with_newline:.2f}")

# 1 - total_bits_with_prompt_with_newline / total_bits_no_prompt_with_newline
# #%%
# total_bits_no_prompt_with_newline / total_bits_no_prompt_no_newline
# #%%
# (total_bits_with_prompt_with_newline - total_bits_with_prompt_no_newline) / 500
#%%
# total_bits / sum(len(tokens) for tokens in phrase_tokens)
# TinyLlama: 5.0 bits per token
# gpt2-xl: 4.6 bits per token
# gpt2-xl: 1.263 bits per character
# TinyLlama: 1.3082 bits per character
#%%
def get_last_token_length(s):
    token_ids = tokenizer.encode(s)
    if len(token_ids)==1:
        return 0
    if len(token_ids)==2:
        return len(s)
    last_token_id = token_ids[-1]
    last_token = tokenizer.convert_ids_to_tokens(last_token_id)
    return len(last_token)
import string
# Filter out special tokens and sort alphabetically
# limit to lower case tokens
# replace ▁ with a space
clean_vocab = {
    token.replace('▁', ' '): id for token, id in tokenizer.vocab.items()
    if token not in tokenizer.all_special_tokens 
    and all(c in string.ascii_lowercase or c == '▁' for c in token)
}
clean_vocab_to_old_id = {k: v for k, v in sorted(clean_vocab.items())}
clean_ids = np.array(list(clean_vocab_to_old_id.values()))
clean_vocab_to_clean_id = {token: i for i, token in enumerate(clean_vocab)}
clean_tokens = list(clean_vocab_to_old_id)
old_ids = np.array(list(clean_vocab_to_old_id.values()))
import bisect
def get_prefix_range(prefix, tokens):
    if prefix == '':
        return 0, len(tokens)
    next_prefix = prefix[:-1] + chr(ord(prefix[-1]) + 1)
    return bisect.bisect_left(tokens, prefix), bisect.bisect_left(tokens, next_prefix)
# dummy_tokens = ['ochastic', 'oci', 'ocia', 'ocity', 'ock', 'ocker', 'ocket', 'ockey', 'oco', 'ocoa']
# get_prefix_range('em', clean_tokens), get_prefix_range('en', clean_tokens), clean_tokens[11423], clean_tokens[11424], clean_tokens[11425]
clean_tokens_set = set(clean_tokens)
clean_token_beginnings = {token[:i] for token in clean_tokens for i in range(len(token)+1)}
clean_token_fragments = {token_beginning[i:] for token_beginning in clean_token_beginnings for i in range(len(token_beginning))}
clean_token_fragments.add('')
prefix_range_precomp = {token_beginning: get_prefix_range(token_beginning, clean_tokens) for token_beginning in clean_token_beginnings}
prefix_slice_precomp = {k: slice(*v) for k, v in prefix_range_precomp.items()}
# write prefix_range_precomp to a json file as {word: [start, end]}
import json
with open('prefix_range_precomp.json', 'w') as f:
    json.dump(prefix_range_precomp, f)

#%%
# len(prefix_range_precomp), len(prefix_slice_precomp), len(clean_tokens), len(clean_vocab_to_old_id), len(clean_vocab_to_clean_id), len(clean_vocab)
# okayay' in clean_token_beginnings
# ' least' in prefix_range_precomp

# %%
# experiments:
#  for a prefix of 300 words,
#    without cache it is 100ms
#    with cache it is 20ms + 6ms to build the previous_kv from the cache
async def process_input_texts(input_texts, prefix_cache_trie=None, batch_texts_are_siblings=False,  tokenizer=tokenizer, model=model):
    # print(f"processing {len(input_texts)} texts")
    # for text in input_texts:
    #     print('\t', text)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    batch_size = len(input_texts)
    n_layers = 22
    n_attn_heads = 4
    head_dimension = 64
    
    # Ensure model and tokenizer are initialized
    if tokenizer is None or model is None:
        initialize_model_and_tokenizer()

    # Encode input and get logits
    inputs = tokenizer(input_texts, return_tensors="pt")
    inputs.to(device)
    input_ids = inputs.input_ids
    # if "gpt" in model_name:
    #     prepend_tokens = torch.full((batch_size, 1), tokenizer.bos_token_id, device="cuda", dtype=torch.long)
    #     input_ids = torch.cat([prepend_tokens, input_ids], dim=1)
    #     if input_ids.dtype != torch.long:
    #         print("input_ids.dtype", input_ids.dtype)
    #         input_ids = input_ids.long()
    #     inputs.input_ids = input_ids
    #     inputs['input_ids'] = input_ids
    #     attention_mask = torch.ones((batch_size, input_ids.shape[1]), device="cuda", dtype=torch.long)
    #     inputs.attention_mask = attention_mask
    #     inputs['attention_mask'] = attention_mask
    assert all(len(input_ids[i]) == len(input_ids[0]) for i in range(batch_size))
    seq_len = len(input_ids[0])
    if batch_texts_are_siblings:
        # input texts should be the same except for the last token
        assert all((input_ids[i][:-1] == input_ids[0][:-1]).all() for i in range(1, batch_size))
    with torch.no_grad():
        # create a trie to cache each previous token's past keys/values
        if prefix_cache_trie is not None:
            failed_to_build_cache = False
            # initialize past keys/values for each layer
            past_keys = torch.empty(n_layers, batch_size, n_attn_heads, seq_len-1, head_dimension, device=device)
            past_values = torch.empty(n_layers, batch_size, n_attn_heads, seq_len-1, head_dimension, device=device)
            leaf_nodes = [None for _ in range(batch_size)]
            # for each text
            node = prefix_cache_trie
            start_cache_build_time = time.time()
            for i in range(batch_size):
                if batch_texts_are_siblings and i > 0:
                    past_keys[:,i,:,:,:] = past_keys[:,0,:,:,:]
                    past_values[:,i,:,:,:] = past_values[:,0,:,:,:]
                    leaf_nodes[i] = leaf_nodes[0]
                    continue
                # for each token
                for t, token in enumerate(input_ids[i][:-1]):
                    token = int(token)
                    if token not in node["children"]:
                        failed_to_build_cache = True
                        break
                    node = node["children"][token]
                    # todo: it might be faster if we changed the dim order to keep things contiguous
                    past_keys[:,i,:,t,:] = node["past_keys"]
                    past_values[:,i,:,t,:] = node["past_values"]
                leaf_nodes[i] = node
            end_cache_build_time = time.time()
            # run the model
            if not failed_to_build_cache:
                print(f"successfully built cache in {end_cache_build_time - start_cache_build_time:.4f} seconds")
                past_key_values = [(past_keys[l], past_values[l]) for l in range(n_layers)]
                attention_mask = torch.ones(batch_size, seq_len-1, device=device)
                start_time = time.time()
                output = model(input_ids=input_ids[:,-1:], attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True);
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                print(f"model inference time: {end_time - start_time:.4f} seconds")
            else:
                print("warning: no cache")
                start_time = time.time()
                output = model(**inputs);
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                print(f"model inference time: {end_time - start_time:.4f} seconds")
                # update the cache from the root
                print("updating cache from root")
                for i in range(batch_size):
                    node = prefix_cache_trie
                    # prev_prior_ill = None;
                    for t, token in enumerate(input_ids[i][:-1]):
                        token = int(token)
                        if token not in node["children"]:
                            node["children"][token] = {"children": {}}
                        node = node["children"][token]
                        node["past_keys"] = torch.stack([output.past_key_values[l][0][i,:,t,:] for l in range(n_layers)]).to(device)
                        node["past_values"] = torch.stack([output.past_key_values[l][1][i,:,t,:] for l in range(n_layers)]).to(device)
                        # node["prior_ill"] = prev_prior_ill + float(output.logits[i, t-1].softmax(-1)[token].log()) if t > 0 else 0.0
                        # prev_prior_ill = node["prior_ill"]
                    leaf_nodes[i] = node
            # update the cache
            for i in range(batch_size):
                last_token = input_ids[i][-1]
                last_token = int(last_token)
                node = leaf_nodes[i] 
                node["children"][last_token] = {"children": {}}
                node["children"][last_token]["past_keys"] = torch.stack([output.past_key_values[l][0][i,:,-1,:] for l in range(n_layers)]).to(device)
                node["children"][last_token]["past_values"] = torch.stack([output.past_key_values[l][1][i,:,-1,:] for l in range(n_layers)]).to(device)
                # node["children"][last_token]["prior_ill"] = node["prior_ill"] + float(output.logits[i, -2].softmax(-1)[last_token].log())
        # run the model without cache
        else:
            start_time = time.time()
            output = model(**inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
        # print(f"Model inference time: {time.time() - start_time:.4f} seconds")
    results = []
    for i in range(batch_size):
        logits = output.logits[i].detach().cpu().numpy()
        # logits -= logits.max(axis=-1, keepdims=True)
        # logits = np.exp(logits)
        # logits /= logits.sum(axis=-1, keepdims=True)
        
        # 

        # Get the logits for the last token
        last_token_logits = logits[-1, :]

        # stop token logits
        stop_token = 2
        stop_token_logit = last_token_logits[stop_token]
        newline_token = 13
        newline_token_logit = last_token_logits[newline_token]

        # safe normalize the logits to get probabilities
        # use float64 since we want to represent small deltas in np.cumsum
        clean_last_token_logits = last_token_logits[clean_ids].astype(np.float64)
        clean_last_token_probs = clean_last_token_logits.copy()
        max_logit = max(clean_last_token_logits.max(), stop_token_logit, newline_token_logit)
        clean_last_token_probs = np.exp(clean_last_token_probs - max_logit)
        stop_token_prob = np.exp(stop_token_logit - max_logit)
        newline_token_prob = np.exp(newline_token_logit - max_logit)
        Z = clean_last_token_probs.sum() + stop_token_prob + newline_token_prob
        # normalize
        clean_last_token_probs /= Z
        stop_token_prob /= Z
        newline_token_prob /= Z
        STOP_PROB = stop_token_prob + newline_token_prob
        # cumulative sum of probabilities
        clean_last_token_probs_cumulative = np.cumsum(clean_last_token_probs)

        result = (np.log(clean_last_token_probs), clean_last_token_probs_cumulative, np.log(STOP_PROB))
        if prefix_cache_trie is not None:
            node = leaf_nodes[i]["children"][int(input_ids[i][-1])]
            node["result"] = result
        results.append(result)
    return results

pc_trie = {"children": {}}
asyncio.run(process_input_texts(['hello'], prefix_cache_trie=pc_trie))
asyncio.run(process_input_texts(['hello hello', 'hello okay'], prefix_cache_trie=pc_trie, batch_texts_are_siblings=True))
asyncio.run(process_input_texts([''], prefix_cache_trie=pc_trie, batch_texts_are_siblings=True))
# tokenizer.encode('')

# %%
# Simple decoding vs past_key_values caching
# experiments:
#  with a prefix of 12 words it is 150ms vs 100ms
#  with a prefix of 120 words it is 1000ms vs 100ms
#  this makes sense if reading is 10-20x faster than writing
# processing 5 different final tokens instead of 1 only moves from 100ms to 130ms
# so we want to do two things:
# - maintain a trie to cache the key values
# - select up to 5 most likely siblings to run at the same time
# the above results are cpu
# moving to gpu moves times from ~100ms to ~10ms
# there is no longer a significant speedup for precomp (<20%)
# (this is likely because I wasn't forcing torch.cuda.synchronize() to accurately measure)
# there is very little overhead of larger batches (~0%)
# (may be the same issue)

#%%
################ SERVER HANDLES TOKEN REQUEST PRIORITIZATION

#%%
def update_from_new_timer_likelihoods(letter_trie, timer_likelihoods, prompt):
    old_registry = set(letter_trie.keys())
    old_vals = [x for x in timer_likelihoods if x in old_registry]
    new_vals = [x for x in timer_likelihoods if x not in old_registry]
    for val in sorted(new_vals, key=lambda x: len(x)): # topologically ordered
        # find the ancestor which appears in the original trie
        trie_ancestor_val = None
        possible_ancestors = []
        for i in range(0, len(val)):
            proposed_ancestor_val, suffix = val[:i], val[i:]
            if suffix in clean_token_beginnings:
                possible_ancestors.append(proposed_ancestor_val)
            if proposed_ancestor_val in old_registry:
                trie_ancestor_val = proposed_ancestor_val
        if trie_ancestor_val is None:
            raise ValueError(f"val {val} does not have an ancestor in the original trie")
        parent_val, letter = val[:-1], val[-1]
        parent = letter_trie[parent_val]
        parent["child_letters"] += letter
        likelihood = timer_likelihoods[val]
        last_token_length = get_last_token_length(prompt + val)
        if last_token_length > len(val):
            print(f"warning: last_token_length {last_token_length} > len(val) {len(val)} for {val}. This can occur when using a prompt.")
            token_ancestor = None
        else:
            token_ancestor = val[:-last_token_length]
        letter_trie[val] = {"likelihood": letter_trie[trie_ancestor_val]["likelihood"] + likelihood, "child_letters": "", "possible_ancestors": possible_ancestors, "token_ancestor": token_ancestor}
    for val in old_vals:
        push_likelihood(letter_trie, timer_likelihoods, val, timer_likelihoods[val])
        

def push_likelihood(letter_trie, timer_likelihoods, val, likelihood):
    # push likelihood to descendants which were not visible (i.e. present in timer_likelihoods)
    node = letter_trie[val]
    node["likelihood"] += likelihood
    for child_val in [val+letter for letter in node["child_letters"]]:
        if child_val not in timer_likelihoods:
            push_likelihood(letter_trie, timer_likelihoods, child_val, likelihood)

def set_mdl(letter_trie):
    # 1. set mdl for immediate ancestors
    for val, node in sorted(letter_trie.items(), key=lambda x: len(x[0])):  # topologically ordered
        node["mdl"] = node["likelihood"]  # overwrite previous mdl with new likelihood
        for ancestor_val in node["possible_ancestors"]:
            ancestor_node = letter_trie[ancestor_val]
            ancestor_node["mdl"] = max(ancestor_node["mdl"], node["mdl"])
    # 2. bubble up mdl
    for val, node in sorted(letter_trie.items(), key=lambda x: len(x[0]), reverse=True):  # topologically REVERSE ordered
        if node["token_ancestor"] is not None:
            ancestor_node = letter_trie[node["token_ancestor"]]
            ancestor_node["mdl"] = max(ancestor_node["mdl"], node["mdl"])
        else:
            if not val == "":
                print(f"warning: val {val} does not have a token_ancestor. this is likely because you are using a prompt.")

def set_larr(letter_trie, val, ancestor_val, ancestor_larr):
    if not val in letter_trie:
        print(f"setting larr for {val} not in trie")
        return
    node = letter_trie[val]
    suffix = val[len(ancestor_val):]
    start, end = prefix_range_precomp[suffix]
    ancestor_larr[start:end] = node["likelihood"]
    if suffix in clean_tokens_set:
        ancestor_larr[start] = node["mdl"]
    for l in node["child_letters"]:
        child_val = val + l
        if suffix+l in clean_token_beginnings:
            set_larr(letter_trie, child_val, ancestor_val, ancestor_larr)


#%% SERVER
from fastapi import FastAPI, WebSocket
import json
import asyncio
import starlette.websockets

app = FastAPI()

# in general the values of the token_trie and priority_queue are the true complete strings
# but the letter_trie, like the one on the client's side, only refers to the text after the prompt
import heapq
async def visit_token_trie_node(letter_trie, priority_queue, cached_results, lm_prefix_cache_trie, node_val, node_prior_ill, node_likelihood, siblings, mutable_prompt, websocket = None):
    # print("Visiting [", node_val, "] with prior ", node_prior_ill, " and likelihood ", node_likelihood, sep='')
    prompt = ''.join(mutable_prompt)
    larr = np.full(len(clean_tokens), node_likelihood)
    node_val_minus_prompt = node_val[len(prompt):]
    if node_val_minus_prompt in letter_trie:
        # print("setting larr for", node_val_minus_prompt)
        set_larr(letter_trie, node_val_minus_prompt, node_val_minus_prompt, larr)
    if node_val not in cached_results:
        # print("Requesting [", node_val, "] from lm", sep='')
        # Get 5 best siblings not already processed, using highest posts (negative priorities)
        max_siblings = 5
        best_siblings = [s for s in siblings if s not in cached_results][:5]
        if not node_val in best_siblings:
            # TODO: why does this ever occur?
            print(f"warning: node_val {node_val_minus_prompt} not in best_siblings: {[x[len(prompt):] for x in best_siblings]}")
        # add node_val to best_siblings if not already present
        best_siblings = [node_val] + [x for x in best_siblings if x != node_val]
        print(f"fetching node_val: {node_val_minus_prompt} with siblings: {[x[len(prompt):] for x in best_siblings]}")

        
        # Tokenize each sibling separately to handle ragged inputs
        sibling_token_lists = [tokenizer(s).input_ids for s in best_siblings]
        token_lengths = [len(tokens) for tokens in sibling_token_lists]
        # Check all sequences have same length
        if not all(length == token_lengths[0] for length in token_lengths):
            print(f"warning: siblings have different token counts {list(zip([x[len(prompt):] for x in best_siblings], token_lengths))}")
            best_siblings = [node_val]
        else:
            # Convert to tensor for prefix comparison
            sibling_inputs = tokenizer(best_siblings, return_tensors="pt") 
            sibling_input_ids = sibling_inputs.input_ids
            batch_size = len(best_siblings)
            # Check all sequences are same except last token
            if not all((sibling_input_ids[i][:-1] == sibling_input_ids[0][:-1]).all() for i in range(1, batch_size)):
                print(f"warning: siblings have different prefixes {[x[len(prompt):] for x in best_siblings]}")
                best_siblings = [node_val]

        results = await process_input_texts(best_siblings, prefix_cache_trie=lm_prefix_cache_trie, batch_texts_are_siblings=True)
        r = results[0]
        probs, cum, stop_prob = r
        cum = cum.astype(float).tolist()
        stop_prob = float(stop_prob)
        response = {'type': 'processed', 'ftp': node_val, 'cum': cum, 'stop_prob': stop_prob, 'prior_ill': node_prior_ill}
        try:
            await websocket.send_text(json.dumps(response))
        except starlette.websockets.WebSocketDisconnect:
            print("WebSocket connection closed... not sending result")
            return False
        except Exception as e:
            print("WebSocket connection closed... not sending result", e)
            return False
        await asyncio.sleep(0.1)  # TODO: can we yield for 0ms instead of 1ms?
        cached_results[node_val] = {'result': r, 'sent': True}
        for sibling, sib_result in zip(best_siblings[1:], results[1:]):
            cached_results[sibling] = {'result': sib_result, 'sent': False}
        print(f"cached_results.keys(): {[x[len(prompt):] for x in cached_results.keys()]}")

    if cached_results[node_val]['sent'] == False:
        print(f"sending node computed as a sibling in a batch: {node_val[len(prompt):]}")
        r = cached_results[node_val]['result']
        probs, cum, stop_prob = r
        cum = cum.astype(float).tolist()
        stop_prob = float(stop_prob)
        response = {'type': 'processed', 'ftp': node_val, 'cum': cum, 'stop_prob': stop_prob, 'prior_ill': node_prior_ill}
        try:
            await websocket.send_text(json.dumps(response))
        except starlette.websockets.WebSocketDisconnect:
            print("WebSocket connection closed... not sending result")
            return False
        cached_results[node_val]['sent'] = True

    r = cached_results[node_val]['result']
    probs = r[0]
    priors = node_prior_ill + probs
    posts = priors + larr
    K = 100  #
    best_token_idxs = np.argpartition(posts, -K)[-K:]
    siblings = sorted([(node_val+clean_tokens[idx], -posts[idx]) for idx in best_token_idxs], key=lambda x: x[1])
    siblings = [s[0] for s in siblings]
    for idx in best_token_idxs:
        token = clean_tokens[idx]
        new_node_val = node_val + token
        new_node_likelihood = None if new_node_val in letter_trie else larr[idx]
        new_node_prior_ill = priors[idx]
        priority = -posts[idx]
        heapq.heappush(priority_queue, (priority, new_node_val, new_node_prior_ill, new_node_likelihood, siblings))
    return True

MAX_RESPONSES_PER_GESTURE = 30
async def process_queue(queue, letter_trie, cached_results, lm_prefix_cache_trie, mutable_prompt, num_processed_this_iter, websocket = None):
    while queue:
        if num_processed_this_iter[0] == MAX_RESPONSES_PER_GESTURE:
            # spin until the next gesture
            await asyncio.sleep(0.001)
            continue
        priority, node_val, node_prior_ill, node_likelihood, siblings = heapq.heappop(queue)
        if websocket.client_state != starlette.websockets.WebSocketState.CONNECTED:
            break
        conn_open = await visit_token_trie_node(
            letter_trie=letter_trie,
            priority_queue=queue,
            cached_results=cached_results,
            lm_prefix_cache_trie=lm_prefix_cache_trie,
            node_val=node_val,
            node_prior_ill=node_prior_ill,
            node_likelihood=node_likelihood,
            siblings=siblings,
            mutable_prompt=mutable_prompt,
            websocket=websocket
        )
        num_processed_this_iter[0] += 1
        await asyncio.sleep(0.001)  # yield
        if not conn_open:
            print("WebSocket connection closed... stopping")
            break
    else:
        raise RuntimeError("Queue Empty")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    initial_letter_trie = {"": {"likelihood": 0, "child_letters": "", "possible_ancestors": [], "token_ancestor": None, "mdl": 0}}
    import copy
    letter_trie = copy.deepcopy(initial_letter_trie)
    cached_results = {}
    lm_prefix_cache_trie = {"children": {}}
    mutable_prompt = []
    mutable_prompt[:] = ''
    queue = [(0, ''.join(mutable_prompt), 0, 0, {'': 0})]
    num_processed_this_iter = [0]
    asyncio.create_task(process_queue(
        queue=queue,
        letter_trie=letter_trie,
        cached_results=cached_results,
        lm_prefix_cache_trie=lm_prefix_cache_trie,
        mutable_prompt=mutable_prompt,
        num_processed_this_iter=num_processed_this_iter,
        websocket=websocket
    ))

    while True:
        # Receive message from client
        try:
            data = await websocket.receive_text()
            message = json.loads(data)
            print("Received message", message)
        except starlette.websockets.WebSocketDisconnect:
            print("WebSocket connection closed")
            break

        if message['type'] == 'reset':
            hard_prompt = message['prompt']
            username = message['username']
            # lookup all username's records in the log and send them to the client
            try:
                with open('log.txt', 'r') as f:
                    lines = []
                    for line in f:
                        x = json.loads(line)
                        if x.get('username') == username:
                            lines.append(x)
            except FileNotFoundError:
                lines = []
            await websocket.send_text(json.dumps({'type': 'log_info', 'content': lines}))
            #
            letter_trie.clear()
            letter_trie.update(copy.deepcopy(initial_letter_trie))
            cached_results.clear()
            lm_prefix_cache_trie.clear()
            lm_prefix_cache_trie.update({"children": {}})
            mutable_prompt[:] = hard_prompt
            queue[:] = [(0, hard_prompt, 0, 0, {hard_prompt: 0})]
            num_processed_this_iter[:] = [0]

        elif message['type'] == 'timer_likelihoods':
            await asyncio.sleep(0.0)  # simulate network latency
            if 'letter_trie' not in locals():
                print("letter_trie not initialized; please call reset first.")
                continue
            assert 'hard_prompt' in locals() and 'queue' in locals()
            start_time = time.time()
            timer_likelihoods = message['content']
            timer_likelihoods = {prefix: v['likelihood'] for prefix, v in timer_likelihoods.items()}
            update_from_new_timer_likelihoods(letter_trie, timer_likelihoods, hard_prompt)
            set_mdl(letter_trie)
            queue[:] = [(0, hard_prompt, 0, 0, {hard_prompt: 0})]  # mutate in place
            num_processed_this_iter[:] = [0]
            print("RESET QUEUE")
            end_time = time.time()
            print(f"update_from_new_timer_likelihoods took {end_time - start_time:.4f} seconds")

        elif message['type'] == 'log':
            with open('log.txt', 'a') as f:
                f.write(json.dumps(message['content']) + '\n')
        #
        # elif message['type'] == 'test':
        #     print("testing")
        #     # await websocket.send_text(json.dumps({'type': 'test', 'content': 10}))
        #     # continue
        #     dummy_prefix_cache_trie = {"children": {}}
        #     results = await process_input_texts([''], prefix_cache_trie=dummy_prefix_cache_trie, batch_texts_are_siblings=False)
        #     result = results[0]
        #     probs, cum, stop_prob = result
        #     probs = probs.astype(float).tolist()  # Convert numpy array to Python list
        #     cum = cum.astype(float).tolist()  # Convert numpy array to Python list
        #     stop_prob = float(stop_prob)
        #     response = {'type': 'test', 'content': {'probs': probs, 'cum': cum, 'stop_prob': stop_prob}}
        #     await websocket.send_text(json.dumps(response))
        elif message['type'] == 'ping':
            await websocket.send_text(json.dumps({'type': 'pong', 'pingTime': message['pingTime']}))
        else:
            print(f"unknown message type: {message['type']}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)

#%%
tokens = tokenizer.encode('this is a test\nthat is a nest')
[tokenizer.decode(token) for token in tokens]