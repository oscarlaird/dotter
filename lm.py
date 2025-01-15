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
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # A small Llama model from Hugging Face
tokenizer = None
model = None
cache = None

def initialize_model_and_tokenizer():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.cuda()
initialize_model_and_tokenizer()


# %%
import string
# Filter out special tokens and sort alphabet
# limit to lower case tokens
# replace ▁ with a space
clean_vocab = {
    token.replace('▁', ' '): id for token, id in tokenizer.vocab.items()
    if token not in tokenizer.all_special_tokens 
    and all(c in string.ascii_lowercase or c == '▁' for c in token)
}
clean_vocab = {k: v for k, v in sorted(clean_vocab.items())}
clean_vocab_to_clean_id = {v: k for k, v in enumerate(clean_vocab)}

print(len(clean_vocab))
print(list(clean_vocab.items())[0:])
# numpy array of ids
clean_ids = np.array(list(clean_vocab.values()))
clean_ids


# %%
list(clean_vocab.keys())
# dump to clean_tokens_str.json
import json
with open('clean_tokens_str.json', 'w') as f:
    json.dump(list(clean_vocab.keys()), f)
# build partial suffixes (any beginning prefix of a token) and dump to partial_suffixes.json
partial_suffixes = {}
for token in clean_vocab:
    for i in range(len(token)):
        partial_suffixes[token[:i + 1]] = True
with open('partial_suffixes.json', 'w') as f:
    json.dump(partial_suffixes, f)

# %%
# experiments:
#  for a prefix of 300 words,
#    without cache it is 100ms
#    with cache it is 20ms + 6ms to build the previous_kv from the cache
async def process_input_texts(input_texts, prefix_cache_trie=None, batch_texts_are_siblings=False, cached_results=None, tokenizer=tokenizer, model=model):
    # print(f"processing {len(input_texts)} texts")
    # for text in input_texts:
    #     print('\t', text)
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
    inputs.to("cuda")
    input_ids = inputs.input_ids
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
            past_keys = torch.empty(n_layers, batch_size, n_attn_heads, seq_len-1, head_dimension, device="cuda")
            past_values = torch.empty(n_layers, batch_size, n_attn_heads, seq_len-1, head_dimension, device="cuda")
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
                # print(f"successfully built cache in {end_cache_build_time - start_cache_build_time:.4f} seconds")
                past_key_values = [(past_keys[l], past_values[l]) for l in range(n_layers)]
                attention_mask = torch.ones(batch_size, seq_len-1, device="cuda")
                start_time = time.time()
                output = model(input_ids=input_ids[:,-1:], attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True);
                torch.cuda.synchronize()
                end_time = time.time()
            else:
                print("warning: no cache")
                start_time = time.time()
                output = model(**inputs);
                torch.cuda.synchronize()
                end_time = time.time()
                # update the cache from the root
                print("updating cache from root")
                for i in range(batch_size):
                    node = prefix_cache_trie
                    for t, token in enumerate(input_ids[i][:-1]):
                        token = int(token)
                        if token not in node["children"]:
                            node["children"][token] = {"children": {}}
                        node = node["children"][token]
                        node["past_keys"] = torch.stack([output.past_key_values[l][0][i,:,t,:] for l in range(n_layers)]).to("cuda")
                        node["past_values"] = torch.stack([output.past_key_values[l][1][i,:,t,:] for l in range(n_layers)]).to("cuda")
                    leaf_nodes[i] = node
            # update the cache
            for i in range(batch_size):
                last_token = input_ids[i][-1]
                last_token = int(last_token)
                node = leaf_nodes[i] 
                node["children"][last_token] = {"children": {}}
                node["children"][last_token]["past_keys"] = torch.stack([output.past_key_values[l][0][i,:,-1,:] for l in range(n_layers)]).to("cuda")
                node["children"][last_token]["past_values"] = torch.stack([output.past_key_values[l][1][i,:,-1,:] for l in range(n_layers)]).to("cuda")
        # run the model without cache
        else:
            start_time = time.time()
            output = model(**inputs);
            torch.cuda.synchronize()
            end_time = time.time()
        # print(f"Model inference time: {time.time() - start_time:.4f} seconds")
    results = []
    for i in range(batch_size):
        logits = output.logits[i].detach().cpu().numpy()
        # logits -= logits.max(axis=-1, keepdims=True)
        # logits = np.exp(logits)
        # logits /= logits.sum(axis=-1, keepdims=True)

        # Get the logits for the last token
        last_token_logits = logits[-1, :]

        # stop token logits
        stop_token = 2
        stop_token_logit = last_token_logits[stop_token]

        # safe normalize the logits to get probabilities
        clean_last_token_logits = last_token_logits[clean_ids]
        clean_last_token_probs = clean_last_token_logits.copy()
        max_logit = max(clean_last_token_logits.max(), stop_token_logit)
        clean_last_token_probs -= max_logit
        clean_last_token_probs = np.exp(clean_last_token_probs)
        stop_token_prob = np.exp(stop_token_logit - max_logit)
        Z = clean_last_token_probs.sum() + stop_token_prob
        # normalize
        clean_last_token_probs /= Z
        stop_token_prob /= Z
        # cumulative sum of probabilities
        clean_last_token_probs_cumulative = np.cumsum(clean_last_token_probs)

        result = (np.log(clean_last_token_probs), clean_last_token_probs_cumulative, np.log(stop_token_prob))
        node = leaf_nodes[i]["children"][int(input_ids[i][-1])]
        node["result"] = result
        if cached_results is not None:
            cached_results[input_texts[i]] = node
        results.append(result)
    return results

# Example usage:
# results = asyncio.run(process_input_texts(["a dog a dog a dog a", "a cat a cat a cat a"]))
# probs, cum = results[1]
# cum[clean_vocab_to_clean_id[" dog"]] - cum[clean_vocab_to_clean_id[" dog"]-1], cum[clean_vocab_to_clean_id[" cat"]] - cum[clean_vocab_to_clean_id[" cat"]-1]


def print_cache(cache, depth=0):
    if depth == 0:
        print('--- cache ---')
    for token,child_node in cache.get("children", {}).items():
        print(depth*'  ' + str(token))
        print_cache(child_node, depth+1)

prefix_cache_trie = {"children": {}}
# prefix_cache_trie = None
# start_time = time.time()
# for n in range(50):
#     s = "cat dog " * n
#     print(s[:-1])
#     results = asyncio.run(process_input_texts([s[:-5]], prefix_cache_trie))
#     results = asyncio.run(process_input_texts([s[:-1]], prefix_cache_trie))
#     probs, cum = results[0]
#     catval = cum[clean_vocab_to_clean_id[" dog"]] - cum[clean_vocab_to_clean_id[" dog"]-1]
# end_time = time.time()
# print(end_time - start_time)
# print(catval)
cached_results = {}
results = asyncio.run(process_input_texts([""], prefix_cache_trie, cached_results=cached_results))
results = asyncio.run(process_input_texts(["man"], prefix_cache_trie, cached_results=cached_results))
results = asyncio.run(process_input_texts(["man plan"], prefix_cache_trie, cached_results=cached_results))
prefix_cache_trie["children"][1].keys()

#%%
results = asyncio.run(process_input_texts(["man plan can pan","man plan can pan", "man plan can can"], prefix_cache_trie, batch_texts_are_siblings=True))

#%%
prefix_cache_trie['children']

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


# %%
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
import json
import asyncio
import starlette.websockets

queue_pool = {}
recent_served_requests_pool = {}
prefix_cache_trie = {"children": {}}

app = FastAPI()

class ResponseModel(BaseModel):
    ftp: str
    logits: list

import time

async def process_queue(websocket: WebSocket):
    # todo: queue should not be global; it should be specific to the websocket
    while True:
        queue = queue_pool[websocket]
        if queue:
            await asyncio.sleep(0.001)  # Short delay to yield control
            # Process the first item in the queue
            try:
                first_entry = queue.pop(0)
                first_tokenization_hash = first_entry[2]
                siblings = [first_entry]
                remainder = []
                for entry in queue:
                    key, priority, tokenization_hash = entry
                    if tokenization_hash == first_tokenization_hash and len(siblings) < 5:
                        siblings.append(entry)
                    else:
                        remainder.append(entry)
                queue_pool[websocket] = remainder
                input_texts = [entry[0] for entry in siblings]
            except IndexError:
                print("queue is empty")
                continue
            # add to recent_served_requests so that it won't be put in queue again
            for text in input_texts:
                recent_served_requests_pool[websocket].append((text, time.time()))
            # prune recent_served_requests to only keep the last 10s
            recent_served_requests_pool[websocket] = [r for r in recent_served_requests_pool[websocket] if time.time() - r[1] < 10]
            results = await process_input_texts(input_texts, prefix_cache_trie=prefix_cache_trie, batch_texts_are_siblings=True)
            for i, result in enumerate(results):
                probs, cum, stop_prob = result
                probs = probs.astype(float).tolist()  # Convert numpy array to Python list
                cum = cum.astype(float).tolist()  # Convert numpy array to Python list
                stop_prob = float(stop_prob)
                response = {'type': 'processed', 'ftp': input_texts[i], 'probs': probs, 'cum': cum, 'stop_prob': stop_prob}
                # todo: time this
                try:
                    if websocket.client_state == starlette.websockets.WebSocketState.CONNECTED:
                        await websocket.send_text(json.dumps(response))
                except starlette.websockets.WebSocketDisconnect:
                    return  # stop sending messages if we are disconnected
        else:
            await asyncio.sleep(0.01)  # poll for new requests

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    queue_pool[websocket] = []
    recent_served_requests_pool[websocket] = []

    # Start the queue processing task
    asyncio.create_task(process_queue(websocket))
    
    while True:
        # Receive message from client
        try:
            data = await websocket.receive_text()
            message = json.loads(data)
        except starlette.websockets.WebSocketDisconnect:
            print("WebSocket connection closed")
            break

        if message['type'] == 'set_queue':
            print("setting queue")
            # Set the global queue with the provided list of strings
            q = message['content']
            recent_keys = [k for k,t in recent_served_requests_pool[websocket]]
            queue_pool[websocket] = [(key, priority, hash(tuple(list(tokenization[:-1])))) for key,priority,tokenization in sorted(q, key=lambda x: x[1], reverse=True) if key not in recent_keys]
        elif message['type'] == 'set_trie':
            pass
            # json_trie = message['content']
            # pretty print the trie with indent 2
            # print(json.dumps(json_trie, indent=2))
        elif message['type'] == 'log':
            # write to log.txt
            with open('log.txt', 'a') as f:
                f.write(json.dumps(message['content']) + '\n')
        elif message['type'] == 'test':
            print("testing")
            # await websocket.send_text(json.dumps({'type': 'test', 'content': 10}))
            # continue
            dummy_prefix_cache_trie = {"children": {}}
            results = await process_input_texts([''], prefix_cache_trie=dummy_prefix_cache_trie, batch_texts_are_siblings=False)
            result = results[0]
            probs, cum, stop_prob = result
            probs = probs.astype(float).tolist()  # Convert numpy array to Python list
            cum = cum.astype(float).tolist()  # Convert numpy array to Python list
            stop_prob = float(stop_prob)
            response = {'type': 'test', 'content': {'probs': probs, 'cum': cum, 'stop_prob': stop_prob}}
            await websocket.send_text(json.dumps(response))

        else:
            print(f"unknown message type: {message['type']}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

#%%
# Get the clean_id for 'the'
len(clean_vocab)
bs = set()
for t in clean_vocab:
    for i in range(len(t)):
        bs.add(t[:i])
len(bs)
# 29000 token beginnings
# 19000 proper token beginnings

# %%
