#%%
import string
dummy_nodes = {
    "": 1,
    "a": 1,
    "b": 2,
    "c": 3,
    "ca": 4,
    "cq": 5
}
def get_dummy_trie_node(s):
    return {
        "letter": s[-1] if s else None,
        "val": s,
        "likelihood": dummy_nodes[s],
        "children": [get_dummy_trie_node(s + c) for c in string.ascii_lowercase if (s + c) in dummy_nodes]
    }
root = get_dummy_trie_node("")
import json
print(json.dumps(root, indent=2))
pretty_print = lambda x: print(json.dumps(x, indent=2))

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
import random
def random_string(n):
    return "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=n))
get_last_token_length(' app')
# %timeit random_string(100)
# %timeit get_last_token_length(random_string(100))

# %%
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
# clean_ids
# len(clean_tokens), len(clean_ids)

# %%
import bisect
def get_prefix_range(prefix, tokens):
    if prefix == '':
        return 0, len(tokens)
    next_prefix = prefix[:-1] + chr(ord(prefix[-1]) + 1)
    print(prefix, next_prefix)
    return bisect.bisect_left(tokens, prefix), bisect.bisect_left(tokens, next_prefix)
dummy_tokens = ['ochastic', 'oci', 'ocia', 'ocity', 'ock', 'ocker', 'ocket', 'ockey', 'oco', 'ocoa']
get_prefix_range('em', clean_tokens), get_prefix_range('en', clean_tokens), clean_tokens[11423], clean_tokens[11424], clean_tokens[11425]

#%%
# token: example
# token_beginning: ex
# token_fragment: xamp
clean_token_beginnings = {token[:i] for token in clean_tokens for i in range(len(token)+1)}
clean_token_fragments = {token_beginning[i:] for token_beginning in clean_token_beginnings for i in range(len(token_beginning))}
clean_token_fragments.add('')
prefix_range_precomp = {token_beginning: get_prefix_range(token_beginning, clean_tokens) for token_beginning in clean_token_beginnings}
prefix_slice_precomp = {k: slice(*v) for k, v in prefix_range_precomp.items()}

#%%
len(clean_token_fragments), len(clean_token_beginnings), len(prefix_range_precomp)

#%%
# write prefix_range_precomp to a json file as {word: [start, end]}
import json
with open('prefix_range_precomp.json', 'w') as f:
    json.dump(prefix_range_precomp, f)

#%%
len(clean_token_fragments), len(clean_token_beginnings)
l = []
queue = ['']
import string
def build_trie_bfs(val):
    is_token_beginning = val in clean_token_beginnings
    d = {
        'val': val,
        'letter': val[-1] if val else None,
        'bitmap': {c: val+c in clean_token_fragments for c in string.ascii_lowercase+' '},
        'first_child_idx': len(l)+len(queue)+1,
        'is_token_beginning': is_token_beginning,
        'bounds': prefix_range_precomp[val] if is_token_beginning else None,
        'is_complete_token': val in clean_tokens,
    }
    l.append(d)
    for c in string.ascii_lowercase+' ':
        if val+c in clean_token_beginnings:
            queue.append(val+c)
while queue:
    val = queue.pop(0)
    build_trie_bfs(val)

#%%
for node in l:
    if node['val'] == 'em':
        print(node)

#%%
import struct
def serialize_trie(l, filename):
    with open(filename, "wb") as f:
        for node in l:
            # bounds
            if node['bounds'] is not None:
                f.write(struct.pack("II", *node['bounds']))
            else:
                f.write(struct.pack("II", 0, 0))
            # bitmap
            bitmap = sum(int(node['bitmap'][c]) << i for i, c in enumerate(string.ascii_lowercase + ' '))
            f.write(struct.pack("I", bitmap))
            # first_child_idx
            f.write(struct.pack("i", node['first_child_idx']))
            # flags
            flags = (node['is_token_beginning'] << 0) | (node['is_complete_token'] << 1)
            f.write(struct.pack("I", flags))
            # letter
            f.write(struct.pack("I", ord(node['letter']) if node['letter'] else 0))
serialize_trie(l, "trie.bin")


#%%
# todo: to ensure a consistent state between this server and the client, we should be running the same code on both sides
import heapq
class TokenRequester:
    """ Given a trie representing the state of the program,
    prioritize requests to the language model."""
    def __init__(self, root):
        self.root = root
        self.node_list = self.gather_nodes(root)
        self.registry = {node['val']: node for node in self.node_list}
        for node in self.node_list:
            node['mdl'] = node['likelihood']
            node['prior_ill'] = 0
            node['l_arr'] = np.full(len(clean_tokens), node['likelihood'], dtype=np.float32)
            node['ill_ancestor_distance'] = get_last_token_length(node['val'])
            parent_length = len(node['val']) - node['ill_ancestor_distance']
            node['parent_val'] = node['val'][:parent_length]
            node['last_token'] = node['val'][parent_length:]
            node['possible_token_ancestors'] = list(self.possible_token_ancestors(node))
            # push back likelihood to immediate token ancestors
            for ancestor,suffix in node['possible_token_ancestors']:
                ancestor_node = self.registry[ancestor]
                ancestor_node['mdl'] = max(ancestor_node['mdl'], node['likelihood'])
                ancestor_node['l_arr'][prefix_slice_precomp[suffix]] = node['likelihood']
                # set l_arr[suffix]=-inf to prevent our node appearing twice in the queue
                if suffix in clean_vocab_to_clean_id:
                    ancestor_node['l_arr'][clean_vocab_to_clean_id[suffix]] = -np.inf
        for node in reversed(self.node_list):  # bubble up mdl
            parent_node = self.registry[node['parent_val']]
            parent_node['mdl'] = max(parent_node['mdl'], node['mdl'])
        self.heap = self.get_in_node_priority_list()
        heapq.heapify(self.heap)
        for node in self.node_list:
            if 'probs' in node:
                self.heap = heapq.merge(self.heap, self.get_node_priority_heap(node))

    def get_in_node_priority_list(self):
        priority_list = []
        for node in self.node_list:
            parent_node = self.registry[node['parent_val']]
            if ('probs' in parent_node and not 'probs' in node) or node['val'] == '':
                priority_list.append((-(node['mdl']+node['prior_ill']), node['val']))
        return list(sorted(priority_list, key=lambda x: x[0]))

    def get_node_priority_heap(self, node):
        assert 'probs' in node
        MAX_REQUESTS = 10
        post = node['l_arr'] + node['probs'] + node['prior_ill']
        best_args = np.argpartition(post, -MAX_REQUESTS)[-MAX_REQUESTS:]
        best_tokens = [clean_tokens[i] for i in best_args]
        print(f"Best tokens for {node['val']}: {best_tokens}")
        queue = [(-post[i], node['val']+token) for i,token in zip(best_args, best_tokens)]  # Negate priority for min-heap
        heapq.heapify(queue)
        return queue

    def update(self, val, result):
        probs, cum, stop_prob = result
        if val in self.registry:
            node = self.registry[val]
            node['probs'] = probs
            # 1. add descendants in the trie to the heap
            descendants = self.gather_nodes(node)[1:]  # TODO: we can stop early  # don't include the node itself.
            for descendant in descendants:
                if descendant['parent_val'] == node['val']:
                    suffix = descendant['last_token']
                    assert suffix in clean_vocab_to_clean_id
                    descendant['prior_ill'] = node['prior_ill'] + node['probs'][clean_vocab_to_clean_id[suffix]]
                    priority_tuple = (-(descendant['mdl']+descendant['prior_ill']), descendant['val'])
                    heapq.heappush(self.heap, priority_tuple)
            # 2. add descendants outside of the trie
            self.heap = list(heapq.merge(self.heap, self.get_node_priority_heap(node)))
        else:
            # todo: allow for fetching this token's successors (only 1 token per gesture is too restrictive)
            # is l_arr correct? Yes.
            # This new node doesn't really belong in the trie, but it should have the same likelihood as its parent.
            # parent = ???
            # us = parent['l_arr'][our_idx] + parent['probs'][our_idx] + parent['prior_ill']
            # post = us + probs
            # best_args = np.argpartition(post, -MAX_REQUESTS)[-MAX_REQUESTS:]
            # best_tokens = [clean_tokens[i] for i in best_args]
            # print(f"Best tokens for {node['val']}: {best_tokens}")
            # queue = [(-post[i], node['val']+token) for i,token in zip(best_args, best_tokens)]  # Negate priority for min-heap
            # heapq.heapify(queue)
            # self.heap = list(heapq.merge(self.heap, queue))
            # pass
         
    def __iter__(self):
        while self.heap:
            yield heapq.heappop(self.heap)

    def pop(self):
        return heapq.heappop(self.heap)

    @staticmethod
    def gather_nodes(root):
        nodes = []
        def dfs(node):
            nodes.append(node)
            for child in node['children']:
                dfs(child)
        dfs(root)
        return nodes
    
    @staticmethod
    def possible_token_ancestors(node):
        suffix = ''
        letters = list(node['val'])
        while suffix in clean_token_fragments and letters:
            suffix = letters.pop() + suffix
            if suffix in clean_token_beginnings:
                yield (''.join(letters), suffix)

    @staticmethod
    def pretty_print_tree(node):
        import json
        print(json.dumps(node, indent=2))

import copy
root_with_mdl = copy.deepcopy(root)
token_requester = TokenRequester(root_with_mdl)

#%%
from importlib import reload
import cache_trie

#%%
a = token_requester.pop()
a

#%%
import asyncio
import nest_asyncio
nest_asyncio.apply()
kv_trie = {'children': {}}
cached_results = {}
val = a[1]
result = asyncio.run(cache_trie.process_input_texts([val], kv_trie, True, cached_results, tokenizer, model, clean_ids))
# [clean_tokens[i] for ggi in result[0][0].argpartition(-10)[-10:]]
result

#%%
token_requester.update(val, result[0])

#%%
b = token_requester.pop()
b

#%%
root

#%%
tokenizer.encode('cash')
for i in range(len(tokenizer.encode('cash'))):
    print(tokenizer.decode(tokenizer.encode('cash')[i:]))

# %%
inputs = tokenizer.encode('hello', return_tensors='pt').cuda()  # Empty string isn't valid input, using 'hello' and moving to GPU
outputs = model(inputs)
print(outputs.logits.shape)  # Print shape of output logits
# print the best 10 tokens
best_token_ids = outputs.logits[0, -1].argsort()[-100:]  # Get top 10 tokens from last position
print(best_token_ids)
tokenizer.convert_ids_to_tokens(best_token_ids)
# %%
word = 'ension'.replace(' ','▁')
token_id = tokenizer.vocab[word]
outputs.logits[0, -1, token_id]

#%%
result
#%%
[x for i,x in enumerate(outputs.logits[0, -1, :]) if i in best_token_ids]
