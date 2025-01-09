import torch
import time
import numpy as np

async def process_input_texts(input_texts, prefix_cache_trie, batch_texts_are_siblings, cached_results, tokenizer, model, clean_ids):
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
    
    assert tokenizer is not None and model is not None

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