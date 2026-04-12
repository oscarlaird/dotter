use crate::ROOT_HASH;
use crate::rolling_hash as rh;
use crate::rolling_hash::Hash;
use crate::safe_float::{Float, ZERO};
use crate::symbol::{Symbol, RADIX};

#[derive(Clone)]
pub struct XLUpdateEntry {
    pub likelihood: Float,
    pub symbol: Symbol,
    pub is_leaf: bool,
}

pub type XLUpdate = rh::RHashMap<XLUpdateEntry>;

pub fn new_xlupdate() -> XLUpdate {
    let mut xlupdate = XLUpdate::default();
    xlupdate.insert(ROOT_HASH, XLUpdateEntry {
        likelihood: ZERO,
        symbol: Symbol::Start,
        is_leaf: true,
    });
    xlupdate
}


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

pub fn set_leaf_indicators(l_update: &mut XLUpdate) {
    let mut interior_nodes = rh::RHashSet::default();
    for (&hash, entry) in l_update.iter() {
        if hash != ROOT_HASH {
            let parent_hash = rh::pop_right(hash, entry.symbol.to_byte());
            interior_nodes.insert(parent_hash);
        }
    }
    for (&hash, entry) in l_update.iter_mut() {
        entry.is_leaf = !interior_nodes.contains(&hash);
    }
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
pub fn merge_xl_pair(a: &XLUpdate, b: &XLUpdate) -> XLUpdate {
    merge_many(&[a, b])
}

fn merge_many(l_updates: &[&XLUpdate]) -> XLUpdate {
    // "edge" := n.is_leaf || (p.exists && !n.exists)
    // proceed to the edge, and include in result if any n.exists
    // likelihood = proper_truncated_l + n.likelihood
    // the sum of direct and carried contributions
    debug_assert!(
        l_updates.iter().all(|l_update| well_formed(l_update)),
        "l_updates are not well-formed"
    );
    let mut result = XLUpdate::default();
    struct Frame {
        n_symbol: Symbol,
        p_hash: Hash,
        p_proper_truncated_l: Float,
    }
    let root_frame = Frame {
        n_symbol: Symbol::Start,
        p_hash: 0,
        p_proper_truncated_l: ZERO,
    };
    let mut frames = Vec::new();
    frames.push(root_frame);
    let mut iters = 0;
    while let Some(Frame { n_symbol, p_hash, p_proper_truncated_l }) = frames.pop() {
        assert!({
            iters += 1;
            iters < 500_000
        }, "merge_many: too many iterations");
        let n_hash = rh::append_right(p_hash, n_symbol.to_byte());
        let (n_edge_hit_count, n_proper_truncated_l, n_direct_l, any_n_exists) = l_updates
            .iter()
            .fold((0, p_proper_truncated_l, ZERO, false), |(hits, sum, direct_l, any_n), &l_update| {
                let n_exists = l_update.contains_key(&n_hash);
                let p_exists = l_update.contains_key(&p_hash);
                let hit_edge = l_update.get(&n_hash).map(|e| e.is_leaf).unwrap_or(true);
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
                (hits + (hit_edge as u32), sum + proper_truncated_l_delta, direct_l + direct_l_delta, any_n || n_exists)
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
                p_proper_truncated_l: n_proper_truncated_l,
            });
        }
    }
    result
}