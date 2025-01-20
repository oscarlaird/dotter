function logaddexp(a, b) {
    if (a === -Infinity) {
        return b;
    }
    if (b === -Infinity) {
        return a;
    }
    if (a > b) {
        return a + Math.log(1 + Math.exp(b - a));
    } else {
        return b + Math.log(1 + Math.exp(a - b));
    }
}

function logsubexp(a, b) {
    if (b > a) {
        throw new Error(`logsubexp: b > a for ${a} and ${b}`);
    }
    return a + Math.log(1 - Math.exp(b - a));
}

function F(lm_result, bounds) {
    let start = bounds[0];
    let end = bounds[1];
    if (start == 0) {
        return Math.log(lm_result.cum[end - 1]);
    }
    return Math.log(lm_result.cum[end - 1] - lm_result.cum[start - 1]);
}

// function push_likelihood(node, new_likelihoods) {
//     // must process all visible nodes in reverse topological order
//     let timer_likelihood = new_likelihoods[node.val];
//     let likelihood_this_iteration = -Infinity;
//     let logprob_viz_desc = -Infinity; // conditional probability that the next letter is visible
//     for (let child of node.children) {
//         if (child.is_visible) {
//             let child_likelihood_this_iteration = push_likelihood(child, new_likelihoods);
//             let child_conditional_logprob = child.post_Z - node.post_Z;
//             if (child_conditional_logprob > 0) {
//                 // if (child_conditional_logprob > 1e-6) {
//                 //     throw new Error("child_conditional_logprob is positive for node " + child.val);
//                 // }
//                 child_conditional_logprob = 0;
//             }
//             logprob_viz_desc = logaddexp(logprob_viz_desc, child_conditional_logprob);
//             likelihood_this_iteration = logaddexp(likelihood_this_iteration, child_conditional_logprob + child_likelihood_this_iteration);
//         } else {
//             child.likelihood += timer_likelihood;
//             if (child.has_children) {
//                 child.unpushed_likelihood += timer_likelihood;
//             }
//         }
//     }
//     if (logprob_viz_desc > 0) {
//         // if (logprob_viz_desc > 1e-6) {
//         //     throw new Error("logprob_viz_desc is positive for node " + node.val);
//         // }
//         logprob_viz_desc = 0;
//     }
//     let logprob_NOTviz_desc = Math.log(1 - Math.exp(logprob_viz_desc));
//     likelihood_this_iteration = logaddexp(likelihood_this_iteration, logprob_NOTviz_desc + timer_likelihood);
//     node.likelihood += likelihood_this_iteration;
//     if (node.likelihood === null || node.likelihood === undefined || isNaN(node.likelihood)) {
//         throw new Error(`Invalid likelihood calculation for node "${node.val}":
//             likelihood_this_iteration=${likelihood_this_iteration},
//             logprob_viz_desc=${logprob_viz_desc},
//             timer_likelihood=${timer_likelihood},
//             node=${node}`);
//     }
//     return likelihood_this_iteration;
// }

// function set_viztrie(node, lm, threshold, prefix_range_precomp) {
//     if (node.prior === null || node.prior === undefined) {
//         throw new Error(`EPP Prior is null for node "${node.val}"`);
//     }
//     // set .visible for nodes which exceed the threshold
//     // update priors
//     // nodes are initialized with a possible_ancestors list and an included_ancestors set
//     // for each possible ancestor
//     for (let possible_ancestor of node.possible_ancestors) {
//         // if we have not already updated on it
//         if (!node.included_ancestors.has(possible_ancestor.ancestor)) {
//             // and the probs are available
//             if (possible_ancestor.ancestor in lm) {
//                 // update the prior
//                 // let suffix = node.val.slice(possible_ancestor.length);
//                 // let bounds = prefix_range_precomp[suffix];
//                 // if (!bounds) {
//                     // throw new Error(`No bounds found for suffix "${suffix}"`);
//                 // }
//                 node.prior = logaddexp(node.prior, lm[possible_ancestor.ancestor].prior_ill + F(lm[possible_ancestor.ancestor], possible_ancestor.bounds));
//                 node.included_ancestors.add(possible_ancestor.ancestor);
//             }
//         }
//     }
//     // recalculate the posterior
//     node.post_Z = node.prior + node.likelihood;
//     if (node.post_Z === null || node.post_Z === undefined || isNaN(node.post_Z)) {
//         throw new Error(`Invalid post_Z calculation for node "${node.val}":
//             prior=${node.prior},
//             likelihood=${node.likelihood},
//             post_Z=${node.post_Z}`);
//     }
//     // descend
//     node.is_visible = node.post_Z > threshold;
//     // N.B. if this becomes false, there may be descendants which we won't take the time to flip back to false
//     // So we should always determine the visible set by gathering, not by filtering the entire node set
//     if (node.is_visible) {
//         // push unpushed likelihoods
//         if (node.has_children) {
//             for (let child of node.children) {
//                 child.likelihood += node.unpushed_likelihood;
//                 if (child.has_children) {
//                     child.unpushed_likelihood += node.unpushed_likelihood;
//                 }
//             }
//             node.unpushed_likelihood = 0;
//         } else {
//             // initialize children
//             initialize_children(node, prefix_range_precomp);
//         }
//         // visit children
//         let total_children_post_Z = node.children.reduce((acc, child) => logaddexp(acc, child.post_Z), -Infinity);
//         for (let child of node.children) {
//             // child.likelihood -= (total_children_post_Z - node.post_Z); // normalize children using likelihood
//             set_viztrie(child, lm, threshold, prefix_range_precomp);
//         }
//     }
// }

const MAX_TOKEN_LENGTH = 16;
function get_possible_ancestors(val, prefix_range_precomp) {
    // move backwards so long as we are the end of a token beginning i.e. we are a token fragment
    // TODO: we could use a set of token fragments to stop this earlier
    let possible_ancestors = [];
    for (let i = Math.max(0, val.length - MAX_TOKEN_LENGTH); i < val.length; i++) {
        let ancestor = val.slice(0, i);
        let suffix = val.slice(i);
        if (suffix in prefix_range_precomp) {
            let bounds = prefix_range_precomp[suffix];
            possible_ancestors.push({ancestor, suffix, bounds});
        }
    }
    return possible_ancestors;
}

let root_node = {
    val: '',
    letter: '',
    is_visible: true,
    likelihood: 0,
    unpushed_likelihood: 0,
    children: [],
    has_children: false,
    prior: 0,
    post_Z: 0,
    included_ancestors: new Set(),
    possible_ancestors: [],
}

function initialize_children(node, prefix_range_precomp) {
    node.unpushed_likelihood = 0;
    let children = [];
    for (let c of 'abcdefghijklmnopqrstuvwxyz $') {
        let child_val = node.val + c;
        let child = {
            val: child_val,
            letter: c,
            is_visible: false,
            likelihood: node.likelihood,
            unpushed_likelihood: 0,
            children: [],
            has_children: false,
            prior: -Infinity,
            post_Z: -Infinity,
            included_ancestors: new Set(),
            possible_ancestors: get_possible_ancestors(child_val, prefix_range_precomp),
            initialized: false,
        }
        children.push(child);
    }
    node.children = children;
    node.has_children = true;
}

function pushl_recalc_post_Z_new(node, new_likelihoods) {
    // post_Z = prior + likelihood should always be correct
    // if .ever_parent_of_visible is true, then likelihood may not be valid, but post_Z must be
    let timer_likelihood = new_likelihoods[node.val];
    let has_visible_children = node.children.some(child => child.is_visible);
    if (!has_visible_children) {
        node.likelihood += timer_likelihood;
        node.post_Z += timer_likelihood; // since all descendants' likelihoods increased by this, post_Z is correct
        node.unpushed_likelihood += timer_likelihood;
        return;
    }
    let post_Z = -Infinity;
    for (let child of node.children) {
        if (child.is_visible) {
            pushl_recalc_post_Z_new(child, new_likelihoods);
        } else { // push likelihood to invisible children
            child.likelihood += timer_likelihood;
            child.post_Z += timer_likelihood;
            child.unpushed_likelihood += timer_likelihood;
        }
        post_Z = logaddexp(post_Z, child.post_Z);
    }
    node.likelihood += timer_likelihood;
    node.post_Z = post_Z;
}

function set_viztrie_new(node, lm, threshold, prefix_range_precomp, pDATA) {
    if (!node.initialized) {
        // for each possible ancestor
        for (let possible_ancestor of node.possible_ancestors) {
            // if we have not already updated on it
            if (!node.included_ancestors.has(possible_ancestor.ancestor)) {
                // and the probs are available
                if (possible_ancestor.ancestor in lm) {
                    node.prior = logaddexp(node.prior, lm[possible_ancestor.ancestor].prior_ill + F(lm[possible_ancestor.ancestor], possible_ancestor.bounds));
                    node.included_ancestors.add(possible_ancestor.ancestor);
                }
            }
        }
        if (node.letter === '$') {
            let parent_val = node.val.slice(0, -1);
            if (parent_val in lm) {
                node.prior = lm[parent_val].prior_ill + lm[parent_val].stop_prob;
            }
        }
        node.post_Z = node.prior + node.likelihood;
        node.initialized = true;
    }
    // assume priors have already been set (incl. updating post_Z)
    node.is_visible = node.post_Z - pDATA > threshold;
    if (node.is_visible) {
        // push unpushed likelihoods
        if (node.has_children) {
            for (let child of node.children) {
                child.likelihood += node.unpushed_likelihood;
                child.post_Z += node.unpushed_likelihood;
                child.unpushed_likelihood += node.unpushed_likelihood;
            }
            node.unpushed_likelihood = 0;
        } else {
            // initialize children
            initialize_children(node, prefix_range_precomp);
        }
        // visit children
        for (let child of node.children) {
            set_viztrie_new(child, lm, threshold, prefix_range_precomp, pDATA);
        }
    }
}

function update_prior_new(update_root_val, node, lm, prefix_range_precomp) {
    let suffix = node.val.slice(update_root_val.length);
    let update_prior_contribution;
    if (suffix === '$') {
        update_prior_contribution = lm[update_root_val].prior_ill + lm[update_root_val].stop_prob;
    } else if (suffix === '') {
        update_prior_contribution = -Infinity;
    } else {
        let bounds = prefix_range_precomp[suffix];
        if (!bounds) {
            throw new Error(`No bounds found for suffix "${suffix}"`);
        }
        update_prior_contribution = lm[update_root_val].prior_ill + F(lm[update_root_val], bounds);
    }
    let old_prior = node.prior;
    node.prior = logaddexp(node.prior, update_prior_contribution);
    if (old_prior === -Infinity) {
        // this implies that we could not have had any visible children and thus our likelihood must be still valid
        node.post_Z = node.prior + node.likelihood;
    } else {
        node.post_Z += node.prior - old_prior;
    }
    let has_children_affected_by_update = node.children.some(child => (suffix+child.letter) in prefix_range_precomp);
    // if the children's priors won't change, then we know that our (unknown likelihood) will not change
    // if the likelihood is constant after this node, then redistributing the prior will have no effect
    // NOTE: we can stop on (!node.ever_parent_of_visible || !has_children_affected_by_update) if we are sure to push the prior update later
    // but for now, it is simpler just to descend to the leaves
    let valid_post_Z = (!has_children_affected_by_update) || (!node.has_children);
    if (valid_post_Z) {
        return;
    }
    // if any children's priors have been changed, we must update them in order to compute our own post_Z
    let post_Z = -Infinity;
    for (let child of node.children) {
        // push unpushed likelihoods
        child.likelihood += node.unpushed_likelihood;
        child.post_Z += node.unpushed_likelihood;
        child.unpushed_likelihood += node.unpushed_likelihood;
        // update children affected by the update
        if ((suffix+child.letter in prefix_range_precomp) || (suffix+child.letter === '$')) {
            update_prior_new(update_root_val, child, lm, prefix_range_precomp);
        }
        post_Z = logaddexp(post_Z, child.post_Z);
    }
    node.unpushed_likelihood = 0;
    node.post_Z = post_Z;
}

function get_node_by_val(node, val) {
    if (node.val === val) {
        return node;
    }
    let next_letter = val[node.val.length];
    let child = node.children.find(child => child.letter === next_letter);
    if (!child) {
        return null;
    }
    return get_node_by_val(child, val);
}

function grab_post_Z_new(node, target_val, target_node_old_post_Z) {
    // update our node's post_Z and return its old value (prior to the update)
    if (node.val === target_val) {
        return target_node_old_post_Z;
    }
    let next_letter = target_val[node.val.length];
    let child = node.children.find(child => child.letter === next_letter);
    if (!child) {
        throw new Error(`No child found for letter "${next_letter}" in node "${node.val} when target_val="${target_val}"`);
    }
    let self_old_post_Z = node.post_Z;
    let old_child_post_Z = grab_post_Z_new(child, target_val, target_node_old_post_Z);
    node.post_Z = logsubexp(node.post_Z, old_child_post_Z);
    node.post_Z = logaddexp(node.post_Z, child.post_Z);
    if (isNaN(node.post_Z)) {
        throw new Error(`Invalid post_Z calculation for node "${node.val}":
            old_child_post_Z=${old_child_post_Z},
            child_post_Z=${child.post_Z},
            node_post_Z=${node.post_Z}`);
    }
    return self_old_post_Z;
}

function update_prior_pipeline(trie, val, lm, prefix_range_precomp) {
    let node = get_node_by_val(trie, val);
    if (node === null) {
        console.log("WARNING: update_prior_pipeline: node is null for val", val);
        return;
    }
    let target_node_old_post_Z = node.post_Z;
    update_prior_new(val, node, lm, prefix_range_precomp);
    console.log("Before grab, node.post_Z", node.post_Z, "for val", val, "trie post_Z", trie.post_Z);
    grab_post_Z_new(trie, val, target_node_old_post_Z);
    console.log("After grab, node.post_Z", node.post_Z, "for val", val, "trie post_Z", trie.post_Z);
    if (node.post_Z > trie.post_Z) {
        console.log("WARNING: update_prior_pipeline: node.post_Z > trie.post_Z for val", val, "node.post_Z", node.post_Z, "trie post_Z", trie.post_Z);
        throw new Error("update_prior_pipeline: node.post_Z > trie.post_Z");
    }
}

// function gather_visible_nodes(node, node_list) {
//     if (node.is_visible) {
//         node_list.push(node);
//     }
//     for (let child of node.children) {
//         gather_visible_nodes(child, node_list);
//     }
// }

export {
    // set_viztrie,
    // gather_visible_nodes,
    // push_likelihood,
    logaddexp,
    F,
    root_node,
    get_node_by_val,
    grab_post_Z_new,
    set_viztrie_new,
    update_prior_new,
    pushl_recalc_post_Z_new,
    update_prior_pipeline
};
