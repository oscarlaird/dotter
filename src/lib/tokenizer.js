import { lmF } from "./lm.js";
import cleanTokensStr from "./clean_tokens_str.json";
// construct a function s->idx from cleanTokensStr
const cleanTokensStrToIdx = Object.fromEntries(cleanTokensStr.map((s, idx) => [s, idx]));
const lm_request_queue_max_size = 20;
// construct PS from cleanTokensStr
const PS = new Set();
for (const token of cleanTokensStr) {
    for (let i = 0; i <= token.length; i++) {
        PS.add(token.slice(0, i));
    }
}
const max_ps_len = 16;

function f(LM, s, t) {
    let t_idx = cleanTokensStrToIdx[t];
    if (LM[s] === undefined) {
        console.log(`[f] LM[s] is undefined for s: ${s}`);
        return -Infinity; // TODO: this could use a lower grade model
        // then on update we would need to subtract this probability
    } else if (LM[s].probs[t_idx] === undefined) {
        console.log(`[f] LM[s].probs[t_idx] is undefined for s: ${s}, t: ${t}`);
        return -Infinity; // TODO: this could use a lower grade model
    } else {
        return LM[s].probs[t_idx];
    }
}
function f_stop(LM, s) {
    if (LM[s] === undefined) {
        console.log(`[f_stop] LM[s] is undefined for s: ${s}`);
        return -Infinity; // TODO: this could use a lower grade model
    } else if (LM[s].stop_prob === undefined) {
        console.log(`[f_stop] LM[s].stop_prob is undefined for s: ${s}`);
        return -Infinity;
    } else {
        return LM[s].stop_prob;
    }
}
function F(LM, m, s) {
    if (LM[m] === undefined) {
        console.log(`[F] LM[m] is undefined for m: ${m}`);
        return -Infinity; // TODO: this could use a lower grade model
    } else {
        return Math.log(lmF(s, cleanTokensStr, LM[m].cum));
    }
}



function get_tokenization(s, tokenizer) {
    return tokenizer.encode(s);
}

function arrayStartsWith(array, prefix) {
    return prefix.every((item, index) => array[index] === item);
}
function arrayEquals(array, other) {
    return array.length === other.length && array.every((item, index) => item === other[index]);
}

function can_have_required_ancestor(node, a) {
    if (node === a) {
        return true;
    }
    // Does there exist n=m+s such that
    // m's tokenization contains a's tokenization
    // and s is a valid partial suffix?
    const n = node.val;
    for (let i = n.length; i >= 0 && i >= n.length - max_ps_len; i--) {
        const m = n.slice(0, i);
        const m_node = node.trie.registry[m];
        if (m_node === undefined) {
            break;
        }
        const s = n.slice(i);
        // if (partial_suffix_exists(s) && arrayStartsWith(m_node.tokenization, a.tokenization)) {
        //     return true;
        // }
        // todo: audit correctness
        // as a quick check we just see if the last two tokens of m agree with the same tokens in a
        let start_idx = Math.max(0, m_node.tokenization.length - 2);
        let end_idx = m_node.tokenization.length;
        const m_two = m_node.tokenization.slice(start_idx, end_idx);
        const a_two = a.tokenization.slice(start_idx, end_idx);
        if (partial_suffix_exists(s) && arrayEquals(m_two, a_two)) {
            return true;
        }
    }
    return false;
}

function partial_suffix_exists(s) {
    if (s.length > max_ps_len) {
        // considerably faster than the hash lookup
        return false;
    }
    return PS.has(s);
}

function logaddexp(a, b) {
    // safe logaddexp
    if (a === -Infinity) return b;
    if (b === -Infinity) return a;
    if (a > b) {
        return a + Math.log(1 + Math.exp(b - a));
    }
    return b + Math.log(1 + Math.exp(a - b));
}
function logsumexp(logs) {
    // reduce with logaddexp
    return logs.reduce((a, b) => logaddexp(a, b));
}

function node_to_string(node) {
    return `Node(${node.val})`;
}

function update_trie(node, update_priors, call_level, lm_request_queue_reference, pDATA_LB, LM, visibility_threshold, tokenizer, update_root = null) {
    if (update_root === null) {
        update_root = node;
    }
    const n = node.val;
    if (update_priors) {
        if (!update_root.in_character_model) {
            throw new Error(`trying to update priors, but update_root is not in character model: ${update_root.val}`);
        }
        // we have just received f(n, *) from the language model
        // update the stop_token
        if (call_level == 0) {
            let stop_descendant = node.trie.registry[node.val + "$"];
            if (stop_descendant) {
                stop_descendant.prior = update_root.prior_ill + f_stop(LM, update_root.val);
            }
        }

        if (call_level > 0 && node.letter !== "$") {
            let suffix_since_update_root = n.slice(update_root.val.length);
            if (!partial_suffix_exists(suffix_since_update_root)) {
                // stop recursing once (descendant-update_root) cannot be a partial suffix
                // TODO: is this correct? 
                return;
            }
            node.prior = logaddexp(node.prior, update_root.prior_ill + F(LM, update_root.val, suffix_since_update_root));
            // PRIOR ILL
            if (node.ill_ancestor === update_root.val) {
                node.prior_ill = update_root.prior_ill + f(LM, update_root.val, suffix_since_update_root);
            }
        }
    }
    // post_ill_Z means unnormalized posterior probability of the node being the end of a token in <X>
    // if this is the stop token, don't request it (since we don't want to predict more tokens after a stop token)
    if (!node.in_character_model && node.trie.registry[node.ill_ancestor] && node.trie.registry[node.ill_ancestor].in_character_model && node.letter !== "$") {
        node.post_ill_Z = max_descendant_likelihood(node, node) + node.prior_ill;
        // if already in queue, update priority
        if (lm_request_queue_reference.has(node.val)) {
            lm_request_queue_reference.update(node.val, node.post_ill_Z);
        // if not in queue, and queue is not full, insert
        } else if (lm_request_queue_reference.get_length() < lm_request_queue_max_size) {
            lm_request_queue_reference.insert(node.val, node.post_ill_Z);
        // if not in queue, and queue is full, and we have higher priority than the least priority, remove the least priority and insert
        } else if (lm_request_queue_reference.least_priority_value < node.post_ill_Z) {
            lm_request_queue_reference.remove_lowest_priority();
            lm_request_queue_reference.insert(node.val, node.post_ill_Z);
        }
    }
    // post_Z means unnormalized posterior
    node.post_Z = node.prior + node.likelihood;  // may not be valid if ever_visible_parent
    let post_UB = node.post_Z - pDATA_LB;
    if (post_UB > (visibility_threshold-1) && node.children.length === 0) {
        // expand node
        // don't expand the stop token (c="$")
        if (node.letter === "$") {
            return;
        }
        // node.children = [new_node(node.val + c) for c in 'abcdefghijklmnopqrstuvwxyz ']
        for (const c of "abcdefghijklmnopqrstuvwxyz $") {
            if (node.force_space && c !== " ") {
                continue;
            }
            if ((node.prohibit_space || node.letter === " ") && c === " ") {
                continue;
            }
            const child_val = node.val + c;

            // Split on last space to get before/after space segments
            let child_tokenization;
            if (c === "$") {
                const stop_token = 2;
                child_tokenization = [...node.tokenization, stop_token];
            } else {
                const lastSpaceIndex = child_val.slice(0, -1).lastIndexOf(" ");  // index of last space (before the last character)
                const before_space = lastSpaceIndex === -1 ? "" : child_val.slice(0, lastSpaceIndex);
                const after_space = lastSpaceIndex === -1 ? child_val : child_val.slice(lastSpaceIndex+1);
                // try to use the previously computed tokenization up to the space
                const before_space_node = node.trie.registry[before_space];
                const before_space_in_registry = before_space.length > 0 && before_space_node !== undefined;
                if (before_space_in_registry) {
                    if (after_space.length > 0 && after_space[0] === " ") {
                        throw new Error(`val has two consecutive spaces! ${child_val}`);
                    }
                    let before_space_tokenization = before_space_node.tokenization;
                    let after_space_tokenization = get_tokenization(after_space, tokenizer).slice(1); // remove <start> special token
                    child_tokenization = [...before_space_tokenization, ...after_space_tokenization]
                } else {
                    child_tokenization = get_tokenization(child_val, tokenizer);
                }
            }
            // console.log("child_tokenization for " + child_val + " is " + child_tokenization);
            const child = {
                val: child_val,
                letter: c,
                likelihood: node.likelihood,
                tokenization: child_tokenization,
                children: [],
                in_character_model: false,
                ever_visible_parent: false,
                is_visible: false,
                ever_visible: false,
                trie: node.trie,
                letter: c,
                parent: node,
                height: node.height + 1,
                // offset: 0,
                timer_fracs: [...node.timer_fracs],
            };
            const a_tokenization = child.tokenization.slice(0, -1);
            for (let i = 1; i <= child.val.length; i++) {
                let a = child.val.slice(0, -i);
                let a_node = node.trie.registry[a];
                if (arrayEquals(a_node.tokenization, a_tokenization)) {
                    child.ill_ancestor = a;
                    child.ill_suffix = child.val.slice(-i);
                    break;
                }
            }
            if (child.ill_ancestor === undefined) {
                throw new Error("Could not find ill_ancestor for: " + child.val);
            }
            // PRIOR
            if (child.letter === "$") {
                if (node.in_character_model) {
                    child.prior = node.prior_ill + f_stop(LM, node.val);
                } else {
                    child.prior = -Infinity;
                }
            } else {
                let prior = -Infinity;
                for (let i = Math.max(0, child.val.length - max_ps_len); i < child.val.length; i++) {
                    // n=m+s
                    const s = child.val.slice(i);
                    if (partial_suffix_exists(s)) {
                        const m = child.val.slice(0, i); // properly shorter than n
                        let m_node = node.trie.registry[m];
                        if (m_node.in_character_model) {
                            prior = logaddexp(prior, m_node.prior_ill + F(LM, m, s));
                        }
                    }
                }
                child.prior = prior;
            }
            if (child.prior === undefined) {
                throw new Error(`child.prior is undefined for: ${child.val}`);
            }
            // PRIOR ILL
            let ill_ancestor_node = node.trie.registry[child.ill_ancestor];
            if (ill_ancestor_node.in_character_model) {
                if (child.letter === "$") {
                    child.prior_ill = ill_ancestor_node.prior_ill + f_stop(LM, child.ill_ancestor);
                } else {
                    child.prior_ill = ill_ancestor_node.prior_ill + f(LM, child.ill_ancestor, child.ill_suffix);
                }
            } else {
                child.prior_ill = -Infinity;
            }
            //
            node.children.push(child);
            node.trie.registry[child.val] = child;
        }
        for (const child of node.children) {
            // we do not need to update priors since we just calculated them
            update_trie(child, false, call_level + 1, lm_request_queue_reference, pDATA_LB, LM, visibility_threshold, tokenizer, update_root);
        }
    } else {
        for (const child of node.children) {
            update_trie(child, update_priors, call_level + 1, lm_request_queue_reference, pDATA_LB, LM, visibility_threshold, tokenizer, update_root);
        }
    }
}

function run_func_w_timing(func, args) {
    const start = performance.now();
    const result = func(...args);
    const end = performance.now();
    console.log(`Function ${func.name} took ${end - start} milliseconds to execute.`);
    return result;
}

function calc_posteriors(trie) {
    // we need to push up posteriors to determine the sizes of parent nodes
    // we could take a top-down, depth-first approach
    // we want a way to avoid visiting all nodes
    // but on each gesture, the post_prob of every node has changed
    // p(n|D) = \sum_l p(n + l|D) which is known for visible l,
    // and for hidden l, they all get the same likelihood
    // so it is possible to calc the posterior for visible nodes
    // only by visiting visible nodes
    // 

    // Process nodes in reverse topological order (bottom-up)
    let nodes = [];
    
    let start = performance.now();
    function collect_nodes(node) {
        nodes.push(node);
        for (const child of node.children) {
            collect_nodes(child);
        }
    }
    collect_nodes(trie);
    console.log(`[calc_posterior] Collecting nodes took ${performance.now() - start} milliseconds`);

    start = performance.now();
    // Reverse topological order - process children before parents
    nodes.reverse();
    console.log(`[calc_posterior] Reversing nodes took ${performance.now() - start} milliseconds`);

    start = performance.now();
    for (const node of nodes) {
        if (node.children.length > 0) {
            // For parent nodes, post_Z is logsumexp of children's post_Z
            node.post_Z = logsumexp(node.children.map(child => child.post_Z));
            
            // Calculate cumulative logsumexp and set y_pos for each child
            let cumsum = -Infinity;
            for (const child of node.children) {
                child.y_relative_bottom = Math.exp(cumsum - node.post_Z);
                cumsum = logaddexp(cumsum, child.post_Z);
                child.y_relative_top = Math.exp(cumsum - node.post_Z);
            }
        }
    }
    console.log(`[calc_posterior] Processing nodes took ${performance.now() - start} milliseconds`);
}

function push_likelihood(node, likelihood, timerFrac) {
    node.likelihood += likelihood;
    node.timer_fracs.push(timerFrac);
    for (const child of node.children) {
        if (!child.is_visible) {
            push_likelihood(child, likelihood, timerFrac);
        }
    }
}

function max_descendant_likelihood(node, required_ancestor) {
    if (!can_have_required_ancestor(node, required_ancestor)) {
        return -Infinity;
    }
    if (!node.ever_visible_parent) {  // all children necessarily have the same likelihood; stop
        return node.likelihood;
   }
   // return max(max_descendant_likelihood(child, required_ancestor) for child in node.children)
   return Math.max(...node.children.map(child => max_descendant_likelihood(child, required_ancestor)));
}

function evidence_lower_bound(node) {
    if (!node.ever_visible_parent) { // likelihood is valid
        return node.likelihood + node.prior;
    }
    // return logsumexp([evidence_lower_bound(child) for child in node.children])
    return logsumexp(node.children.map(child => evidence_lower_bound(child)));
}

export { update_trie, node_to_string, get_tokenization, calc_posteriors, push_likelihood, run_func_w_timing };