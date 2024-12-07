import { lmF } from "./lm.js";
import cleanTokensStr from "./clean_tokens_str.json";
// construct a function s->idx from cleanTokensStr
const cleanTokensStrToIdx = Object.fromEntries(cleanTokensStr.map((s, idx) => [s, idx]));
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
        return -Infinity; // TODO: this could use a lower grade model
    } else if (LM[s].probs[t_idx] === undefined) {
        return -Infinity; // TODO: this could use a lower grade model
    } else {
        return LM[s].probs[t_idx];
    }
}
function F(LM, m, s) {
    if (LM[m] === undefined) {
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

function update_trie(node, recompute_priors, call_level, queue_reference, pDATA_LB, LM, visibility_threshold, tokenizer) {
    const n = node.val;
    if (recompute_priors) {
        // we have just received f(n, *) from the language model
        if (call_level > 0) {
            // we are not the root of the update call
            // evaluate priors from ancestors
            // ILL
            let ill_ancestor_node = node.trie.registry[node.ill_ancestor];
            node.prior_ill = ill_ancestor_node.prior_ill + f(LM, node.ill_ancestor, node.ill_suffix);
            // partial suffixes
            let prior = -Infinity;
            for (let i = Math.max(0, n.length - max_ps_len); i < n.length; i++) {
                // n=m+s
                const s = n.slice(i);
                if (partial_suffix_exists(s)) {
                    const m = n.slice(0, i); // properly shorter than n
                    let m_node = node.trie.registry[m];
                    if (m_node.in_character_model) {
                        prior = logaddexp(prior, m_node.prior_ill + F(LM, m, s));
                    }
                }
            }
            node.prior = prior;
        }
    }
    // post_ill_Z means unnormalized posterior probability of the node being the end of a token in <X>
    if (!node.in_character_model && node.trie.registry[node.ill_ancestor].in_character_model) {
        node.post_ill_Z = max_descendant_likelihood(node, node) + node.prior_ill;
        // request node with priority given by node.post_ill_Z
        queue_reference.push([node.val, node.post_ill_Z]);
    }
    // post_Z means unnormalized posterior
    node.post_Z = node.prior + node.likelihood;  // may not be valid if ever_visible_parent
    let post_UB = node.post_Z - pDATA_LB;
    if (post_UB > visibility_threshold && node.children.length === 0) {
        // expand node
        // node.children = [new_node(node.val + c) for c in 'abcdefghijklmnopqrstuvwxyz ']
        for (const c of "abcdefghijklmnopqrstuvwxyz ") {
            if (node.force_space && c !== " ") {
                continue;
            }
            if ((node.prohibit_space || node.letter === " ") && c === " ") {
                continue;
            }
            const child_val = node.val + c;

            // Split on last space to get before/after space segments
            let child_tokenization;
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

            // console.log("child_tokenization for " + child_val + " is " + child_tokenization);
            const child = {
                val: child_val,
                letter: c,
                likelihood: node.likelihood,
                tokenization: child_tokenization,
                children: [],
                in_character_model: false,
                ever_visible_parent: false,
                trie: node.trie,
                letter: c,
                parent: node,
                height: node.height + 1,
                // dummy period, offset
                period: node.period,
                offset: Math.random() * node.period,
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
            node.children.push(child);
            node.trie.registry[child.val] = child;
        }
        for (const child of node.children) {
            // since this is an expansion and we are evaluating these nodes for the first time,
            // we need to recompute priors (since they have not yet been computed)
            update_trie(child, true, call_level + 1, queue_reference, pDATA_LB, LM, visibility_threshold);
        }
    } else {
        for (const child of node.children) {
            update_trie(child, recompute_priors, call_level + 1, queue_reference, pDATA_LB, LM, visibility_threshold);
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
    // Process nodes in reverse topological order (bottom-up)
    let nodes = [];
    function collect_nodes(node) {
        nodes.push(node);
        for (const child of node.children) {
            collect_nodes(child);
        }
    }
    collect_nodes(trie);

    // Reverse topological order - process children before parents
    nodes.reverse();

    for (const node of nodes) {
        if (node.children.length > 0) {
            // For parent nodes, post_Z is logsumexp of children's post_Z
            node.post_Z = logsumexp(node.children.map(child => child.post_Z));
            
            // Calculate cumulative logsumexp and set y_pos for each child
            let cumsum = -Infinity;
            for (const child of node.children) {
                child.y_relative_bottom = Math.exp(cumsum - node.post_Z);
                cumsum = logsumexp([cumsum, child.post_Z]);
                child.y_relative_top = Math.exp(cumsum - node.post_Z);
            }
        }
    }
}

function clear_visibility(node) {
    node.is_visible = false;
    for (const child of node.children) {
        clear_visibility(child);
    }
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

export { update_trie, node_to_string, get_tokenization, calc_posteriors, clear_visibility, push_likelihood, run_func_w_timing };