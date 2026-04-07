use crate::trie::{ROOT_HASH, INVALID_HASH, ROOT_STRING};
use crate::rolling_hash as rh;
use crate::rolling_hash::Hash;
use crate::safe_float::{Float, ZERO};
use crate::symbol::{Symbol, RADIX};

#[derive(Clone)]
pub(crate) struct LUpdate {
    pub(crate) likelihoods: rh::RHashMap<Float>,
    pub(crate) cpc_form: bool, // complete prefix code form
}

pub struct XLUpdateEntry {
    likelihood: Float,
    symbol: Symbol,
    is_leaf: bool,
}

pub type XLUpdate = rh::RHashMap<XLUpdateEntry>;

fn check_connected(l_update: &XLUpdate) -> bool {
    l_update.iter().all(|(&hash, entry)| {
        hash == ROOT_HASH || {
            let parent_hash = rh::pop_right(hash, entry.symbol.to_byte());
            l_update.contains_key(&parent_hash)
        }
    })
}

fn leaf_indicators_correct(l_update: &XLUpdate) -> bool {
    let mut interior_nodes = rh::RHashSet::default();
    for (&hash, entry) in l_update.iter() {
        if hash != ROOT_HASH {
            let parent_hash = rh::pop_right(hash, entry.symbol.to_byte());
            interior_nodes.insert(parent_hash);
        }
    }
    l_update.iter().all(|(&hash, entry)| {
        entry.is_leaf == !interior_nodes.contains(&hash)
    })
}

fn well_formed(l_update: &XLUpdate) -> bool {
    l_update.contains_key(&ROOT_HASH) && check_connected(l_update) && leaf_indicators_correct(l_update)
}

#[cfg(feature = "proof")]
fn truncate(x: &str, l_update: &XLUpdate) -> Hash {
    for i in (1..=x.len()).rev() {
        let prefix_hash = rh::hash_string(&x[..i]);
        if l_update.contains_key(&prefix_hash) {
            return prefix_hash;
        }
    }
    unreachable!()
}




// correctness theorem: sum_i(l_update_i[truncate(x, lupdate_i)]) = l_merged[truncate(x, l_merged)]
fn merge_many(l_updates: &[&XLUpdate]) -> XLUpdate {
    // "edge" := n.is_leaf || (p.exists && !n.exists)
    // proceed to the edge, and include in result if any n.exists
    // likelihood = proper_truncated_l + n.likelihood
    assert!(l_updates.iter().all(|l_update| well_formed(l_update)));
    let mut result = XLUpdate::default();
    struct Frame {
        n_symbol: Symbol,
        p_hash: Hash,
        p_edge_hit_count: u32,
        p_proper_truncated_l: Float,
    }
    let root_frame = Frame {
        n_symbol: Symbol::Start,
        p_hash: INVALID_HASH,
        p_edge_hit_count: 0,
        p_proper_truncated_l: ZERO,
    };
    let mut frames = Vec::new();
    frames.push(root_frame);
    while let Some(Frame { n_symbol, p_hash, p_edge_hit_count, p_proper_truncated_l }) = frames.pop() {
        let n_hash = rh::append_right(p_hash, n_symbol.to_byte());
        let (n_edge_hit_count, n_proper_truncated_l, n_direct_l, any_n_exists) = l_updates
            .iter()
            .fold((p_edge_hit_count, p_proper_truncated_l, ZERO, false), |(hits, sum, direct_l, any_n), &l_update| {
                let n_exists = l_update.contains_key(&n_hash);
                let p_exists = l_update.contains_key(&p_hash);
                let n_is_leaf = n_exists && l_update.get(&n_hash).unwrap().is_leaf;
                let edge = n_is_leaf || (p_exists && !n_exists);
                let proper_truncated_l_delta = if p_exists && !n_exists {
                    l_update.get(&p_hash).unwrap().likelihood
                } else {
                    ZERO
                };
                let direct_l_delta =  if n_exists {
                    l_update.get(&n_hash).unwrap().likelihood
                } else {
                    ZERO
                };
                (hits + (edge as u32), sum + proper_truncated_l_delta, direct_l + direct_l_delta, any_n || n_exists)
            });
        let is_edge = n_edge_hit_count == l_updates.len() as u32;
        if any_n_exists {
            result.insert(n_hash, XLUpdateEntry {
                likelihood: n_direct_l + n_proper_truncated_l,
                symbol: n_symbol,
                is_leaf: is_edge,
            });
        }
        if is_edge {
            continue;
        }
        for slot in 0..RADIX {
            let child_symbol = Symbol::from_slot(slot);
            frames.push(Frame {
                n_symbol: child_symbol,
                p_hash: n_hash,
                p_edge_hit_count: n_edge_hit_count,
                p_proper_truncated_l: n_proper_truncated_l,
            });
        }
    }
    result
}

struct KernelFrame {
    n_symbol: Symbol,
    p_path: String,
    p_edge_hit_count: u32,
    p_proper_truncated_l: Float,
}

struct KNode {
    path: String,
    parent_path: Option<String>,
    symbol: Symbol,
    is_leaf: bool,
    likelihood: Float,
}

type KUpdate = Vec<KNode>;

struct KernelNodeContrib {
    n_edge_hit_count: u32,
    n_proper_truncated_l: Float,
    n_direct_l: Float,
    any_n_exists: bool,
}

fn symbol_char(symbol: Symbol) -> char {
    symbol.to_byte() as char
}

fn k_contains(update: &KUpdate, path: &str) -> bool {
    for node in update {
        if node.path == path {
            return true;
        }
    }
    false
}

fn k_get_node<'a>(update: &'a KUpdate, path: &str) -> Option<&'a KNode> {
    for node in update {
        if node.path == path {
            return Some(node);
        }
    }
    None
}

fn xl_to_k(update: &XLUpdate) -> KUpdate {
    struct PendingNode {
        hash: Hash,
        symbol: Symbol,
        is_leaf: bool,
        likelihood: Float,
    }
    let mut pending = Vec::with_capacity(update.len());
    for (&hash, entry) in update.iter() {
        pending.push(PendingNode {
            hash,
            symbol: entry.symbol,
            is_leaf: entry.is_leaf,
            likelihood: entry.likelihood,
        });
    }

    let mut hash_paths: Vec<(Hash, String)> = Vec::with_capacity(update.len());
    let mut nodes: KUpdate = Vec::with_capacity(update.len());
    while !pending.is_empty() {
        let mut progress = false;
        let mut i = 0usize;
        while i < pending.len() {
            let p = &pending[i];
            if p.hash == ROOT_HASH {
                let root_path = ROOT_STRING.to_string();
                hash_paths.push((p.hash, root_path.clone()));
                nodes.push(KNode {
                    path: root_path,
                    parent_path: None,
                    symbol: p.symbol,
                    is_leaf: p.is_leaf,
                    likelihood: p.likelihood,
                });
                pending.swap_remove(i);
                progress = true;
                continue;
            }

            let parent_hash = rh::pop_right(p.hash, p.symbol.to_byte());
            let mut parent_path = None;
            for (h, path) in &hash_paths {
                if *h == parent_hash {
                    parent_path = Some(path.clone());
                    break;
                }
            }
            if let Some(parent_path) = parent_path {
                let mut path = parent_path.clone();
                path.push(symbol_char(p.symbol));
                hash_paths.push((p.hash, path.clone()));
                nodes.push(KNode {
                    path,
                    parent_path: Some(parent_path),
                    symbol: p.symbol,
                    is_leaf: p.is_leaf,
                    likelihood: p.likelihood,
                });
                pending.swap_remove(i);
                progress = true;
                continue;
            }
            i += 1;
        }
        assert!(progress, "xl_to_k: unable to resolve parent paths");
    }

    nodes
}

fn k_to_x(update: &KUpdate) -> XLUpdate {
    let mut out = XLUpdate::default();
    for node in update {
        out.insert(
            rh::hash_string(&node.path),
            XLUpdateEntry {
                likelihood: node.likelihood,
                symbol: node.symbol,
                is_leaf: node.is_leaf,
            },
        );
    }
    out
}

fn well_formed_k(update: &KUpdate) -> bool {
    if !k_contains(update, ROOT_STRING) {
        return false;
    }
    for node in update {
        if node.path == ROOT_STRING {
            if node.parent_path.is_some() {
                return false;
            }
            continue;
        }
        if node.parent_path.is_none() {
            return false;
        }
        if !k_contains(update, node.parent_path.as_ref().unwrap()) {
            return false;
        }
    }
    for node in update {
        let mut has_child = false;
        for other in update {
            if other.parent_path.as_deref() == Some(node.path.as_str()) {
                has_child = true;
                break;
            }
        }
        if node.is_leaf != !has_child {
            return false;
        }
    }
    true
}

fn accumulate_node_contrib(
    l_updates: &[&KUpdate],
    n_path: &str,
    p_path: &str,
    p_edge_hit_count: u32,
    p_proper_truncated_l: Float,
) -> KernelNodeContrib {
    let mut n_edge_hit_count = p_edge_hit_count;
    let mut n_proper_truncated_l = p_proper_truncated_l;
    let mut n_direct_l = ZERO;
    let mut any_n_exists = false;

    for &l_update in l_updates {
        let n_node = k_get_node(l_update, n_path);
        let p_node = k_get_node(l_update, p_path);
        let n_exists = n_node.is_some();
        let p_exists = p_node.is_some();
        let n_is_leaf = n_node.map(|node| node.is_leaf).unwrap_or(false);
        let edge = n_is_leaf || (p_exists && !n_exists);

        if edge {
            n_edge_hit_count += 1;
        }
        if p_exists && !n_exists {
            n_proper_truncated_l += p_node.unwrap().likelihood;
        }
        if n_exists {
            n_direct_l += n_node.unwrap().likelihood;
            any_n_exists = true;
        }
    }

    KernelNodeContrib {
        n_edge_hit_count,
        n_proper_truncated_l,
        n_direct_l,
        any_n_exists,
    }
}

fn merge_many_kernel(l_updates: &[&XLUpdate]) -> XLUpdate {
    // Imperative reference kernel for formal verification.
    // Same semantics as `merge_many`, but uses a Vec-backed representation
    // with explicit loops (no iterator combinators, no hash-map algorithmic state).
    assert!(l_updates.iter().all(|l_update| well_formed(l_update)));

    let mut k_updates_storage = Vec::with_capacity(l_updates.len());
    for &l_update in l_updates {
        let k_update = xl_to_k(l_update);
        assert!(well_formed_k(&k_update));
        k_updates_storage.push(k_update);
    }
    let mut k_updates = Vec::with_capacity(k_updates_storage.len());
    for k_update in &k_updates_storage {
        k_updates.push(k_update);
    }

    let mut result = Vec::new();
    let mut frames = Vec::new();
    frames.push(KernelFrame {
        n_symbol: Symbol::Start,
        p_path: String::new(),
        p_edge_hit_count: 0,
        p_proper_truncated_l: ZERO,
    });

    while let Some(KernelFrame { n_symbol, p_path, p_edge_hit_count, p_proper_truncated_l }) = frames.pop() {
        let mut n_path = p_path.clone();
        n_path.push(symbol_char(n_symbol));
        let KernelNodeContrib {
            n_edge_hit_count,
            n_proper_truncated_l,
            n_direct_l,
            any_n_exists,
        } = accumulate_node_contrib(
            &k_updates,
            &n_path,
            &p_path,
            p_edge_hit_count,
            p_proper_truncated_l,
        );

        let is_edge = n_edge_hit_count == l_updates.len() as u32;
        if any_n_exists {
            result.push(KNode {
                path: n_path.clone(),
                parent_path: if p_path.is_empty() { None } else { Some(p_path.clone()) },
                symbol: n_symbol,
                is_leaf: is_edge,
                likelihood: n_direct_l + n_proper_truncated_l,
            });
        }
        if is_edge {
            continue;
        }
        for slot in 0..RADIX {
            frames.push(KernelFrame {
                n_symbol: Symbol::from_slot(slot),
                p_path: n_path.clone(),
                p_edge_hit_count: n_edge_hit_count,
                p_proper_truncated_l: n_proper_truncated_l,
            });
        }
    }

    k_to_x(&result)
}





// OLD STUFF
impl LUpdate {
    pub(crate) fn new() -> Self {
        Self {
            likelihoods: rh::RHashMap::default(),
            cpc_form: false,
        }
    }

    pub(crate) fn deref(&self) -> &rh::RHashMap<Float> {
        &self.likelihoods
    }

    pub(crate) fn deref_mut(&mut self) -> &mut rh::RHashMap<Float> {
        &mut self.likelihoods
    }

    pub(crate) fn to_cpc_form(&mut self) {
        // Complete prefix code form ensures that no node's prefix appears as another node: every sequence represented is maximal/non-overlapping.
        // Example for alphabet {a, b, c}:
        // Starting tree:
        //   root(4.0)
        //     |- a(3.0)
        //         |- aa(2.0)
        //     |- b(1.0)
        // Transformation yields set:
        //   { aa(2.0), ab(3.0), ac(3.0), b(1.0), c(4.0) }
        //   (each string is now maximal/non-prefix of any other in set)
        if self.cpc_form {
            return;
        }
        self.cpc_form = true;
        struct Entry {
            hash: Hash,
            likelihood: Float,
        }
        let mut new_entries: Vec<Entry> = Vec::new();
        let mut remove_entries: Vec<Hash> = Vec::new();
        for (hash, &likelihood) in self.likelihoods.iter() {
            let mut has_any_children = false;
            let mut non_preexisting_child_hashes: Vec<Hash> = Vec::new();
            for slot in 0..RADIX {
                let child_hash = rh::append_right(*hash, Symbol::slot_to_byte(slot));
                if self.likelihoods.contains_key(&child_hash) {
                    has_any_children = true;
                } else {
                    non_preexisting_child_hashes.push(child_hash);
                }
            }
            if !has_any_children {
                continue;
            }
            new_entries.extend(
                non_preexisting_child_hashes
                    .iter()
                    .map(|child_hash| Entry {
                        hash: *child_hash,
                        likelihood,
                    }),
            );
            remove_entries.push(*hash);
        }
        for hash in remove_entries {
            self.likelihoods.remove(&hash);
        }
        self.likelihoods
            .extend(new_entries.into_iter().map(|entry| (entry.hash, entry.likelihood)));
    }

    fn merge_many(l_tries: &[&Self]) -> Self {
        // TODO: it is possible to make this faster with an in-place merge
        // where we stop descent if we haven't hit ourself and we have hit_count=len-1
        // TODO: this function needs to take account of empty likelihood updates
        struct Frame {
            hash: Hash,
            likelihood: Float, // including us
            hit_count: u32, // including us
        }
        // assume that all likelihood updates are already in complete prefix code form i.e. no node is the prefix of another
        for &l_trie in l_tries {
            assert!(
                l_trie.cpc_form,
                "All input tries must be in CPC form before merging"
            );
        }
        let mut result = Self::new();
        let mut walkers = vec![Frame {
            hash: ROOT_HASH,
            likelihood: ZERO,
            hit_count: 0,
        }];
        let mut iters = 0;
        while let Some(Frame {
            hash,
            likelihood,
            hit_count,
        }) = walkers.pop()
        {
            iters += 1;
            assert!(iters < 100_000, "merge likelihood update trie: too many iterations");
            let (hit_delta, likelihood_delta) = l_tries
                .iter()
                .filter_map(|l_trie| l_trie.likelihoods.get(&hash))
                .fold((0, ZERO), |(hits, sum), &v| (hits + 1, sum + v));
            let new_hits = hit_count + hit_delta;
            let new_likelihood = likelihood + likelihood_delta;
            if new_hits == (l_tries.len() as u32) {
                result.likelihoods.insert(hash, new_likelihood);
                continue;
            }
            for slot in 0..RADIX {
                let child_hash = rh::append_right(hash, Symbol::slot_to_byte(slot));
                walkers.push(Frame {
                    hash: child_hash,
                    likelihood: new_likelihood,
                    hit_count: new_hits,
                });
            }
        }
        result.cpc_form = true;
        result
    }

    pub(crate) fn merge(&self, other: &Self) -> Self {
        if self.deref().is_empty() {
            return other.clone();
        }
        if other.deref().is_empty() {
            return self.clone();
        }
        Self::merge_many(&[self, other])
    }
}
