use std::collections::{HashMap, HashSet};

use render_utils::{ExpandedSnapshot, ExpandedSnapshotNode};
use serde::{Deserialize, Serialize};
use timer_spacing::{DEFAULT_MAX_ITER, TimerSpacingParams, constant_phases, optimize};

pub const BOX_WIDTH: f64 = 37.0;
const BOX_WIDTH_CHILDREN_MULTIPLIER: f64 = 1.0;
const ROOT_NODE_WIDTH: f64 = BOX_WIDTH * 1.5;
pub const ROOT_SYMBOL: char = 'A';
pub const SPACE_SYMBOL: char = 'S';
pub const STOP_SYMBOL: char = 'Z';

pub const SCROLL_CENTERING_WEIGHT: f64 = 1.0;
pub const SCROLL_STABILITY_WEIGHT: f64 = 4.0;
pub const SINGLE_PARENT_NODE_WIDTH_PX: f64 = BOX_WIDTH;
pub const SCROLL_TARGET_X_PX: f64 = 400.0;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VisibleNodeTimer {
    pub phase: f64,
}

pub type VisibleNodeTimerMap = HashMap<String, VisibleNodeTimer>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScrollLayoutState {
    pub first_fork_depth: Option<usize>,
    pub first_fork_full_string: Option<String>,
    pub scroll_offset: usize,
    pub scroll_root: String,
    pub scroll_ancestor_keys: Vec<String>,
    pub rendered_node_keys: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VisualNode {
    pub full_string: String,
    pub node: ExpandedSnapshotNode,
    pub symbol: char,
    pub parent_key: Option<String>,
    pub children: Vec<String>,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LikelihoodModel {
    pub mu_delay: f64,
    pub stddev_delay: f64,
    pub outliers: f64,
    pub period: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LikelihoodPayloadNode {
    #[serde(rename = "l")]
    pub likelihood: f64,
    pub phase: f64,
}

fn final_symbol(full_string: &str) -> char {
    if full_string == ROOT_SYMBOL.to_string() {
        ROOT_SYMBOL
    } else {
        full_string.chars().last().unwrap_or(ROOT_SYMBOL)
    }
}

pub fn phrase_to_trie_string(target_phrase: &str) -> String {
    format!(
        "{ROOT_SYMBOL}{}",
        target_phrase.split(' ').collect::<Vec<_>>().join("S")
    )
}

fn node_depth(full_string: &str) -> usize {
    if full_string == ROOT_SYMBOL.to_string() {
        0
    } else {
        full_string.len().saturating_sub(1)
    }
}

fn snapshot_node_passes_threshold(
    node: &ExpandedSnapshotNode,
    root_z: f32,
    expansion_threshold: f32,
) -> bool {
    node.z - root_z > expansion_threshold
}

fn build_tree(snapshot: &ExpandedSnapshot) -> HashMap<String, VisualNode> {
    let mut nodes = HashMap::new();
    for (full_string, node) in snapshot {
        nodes.insert(
            full_string.clone(),
            VisualNode {
                full_string: full_string.clone(),
                node: node.clone(),
                symbol: final_symbol(full_string),
                parent_key: if full_string == &ROOT_SYMBOL.to_string() {
                    None
                } else {
                    Some(full_string[..full_string.len() - 1].to_string())
                },
                children: Vec::new(),
                x: 0.0,
                y: 0.0,
                width: 0.0,
                height: 0.0,
            },
        );
    }
    let keys = nodes.keys().cloned().collect::<Vec<_>>();
    for key in keys {
        let parent_key = nodes.get(&key).and_then(|node| node.parent_key.clone());
        if let Some(parent_key) = parent_key {
            if let Some(parent) = nodes.get_mut(&parent_key) {
                parent.children.push(key.clone());
            }
        }
    }
    for node in nodes.values_mut() {
        node.children.sort();
    }
    nodes
}

fn root_z_of(snapshot: &ExpandedSnapshot) -> f32 {
    snapshot
        .get(&ROOT_SYMBOL.to_string())
        .map(|node| node.z)
        .unwrap_or(0.0)
}

fn ancestor_keys_through_root(full_string: &str) -> Vec<String> {
    if full_string == ROOT_SYMBOL.to_string() {
        return vec![ROOT_SYMBOL.to_string()];
    }
    let mut keys = vec![ROOT_SYMBOL.to_string()];
    for depth in 1..=node_depth(full_string) {
        keys.push(full_string[..depth + 1].to_string());
    }
    keys
}

fn filter_nodes_by_key_set(
    nodes: &HashMap<String, VisualNode>,
    keys: &HashSet<String>,
) -> HashMap<String, VisualNode> {
    let mut filtered = HashMap::new();
    for key in keys {
        if let Some(node) = nodes.get(key) {
            let mut next = node.clone();
            next.children.retain(|child| keys.contains(child));
            filtered.insert(key.clone(), next);
        }
    }
    filtered
}

pub fn build_visible_tree(
    snapshot: &ExpandedSnapshot,
    expansion_threshold: f32,
) -> HashMap<String, VisualNode> {
    let nodes = build_tree(snapshot);
    let root_z = root_z_of(snapshot);
    let mut visible_keys = HashSet::from([ROOT_SYMBOL.to_string()]);
    for (full_string, node) in snapshot {
        if !snapshot_node_passes_threshold(node, root_z, expansion_threshold) {
            continue;
        }
        for ancestor in ancestor_keys_through_root(full_string) {
            visible_keys.insert(ancestor);
        }
    }
    filter_nodes_by_key_set(&nodes, &visible_keys)
}

fn subtree_keys(nodes: &HashMap<String, VisualNode>, root_key: &str) -> HashSet<String> {
    let mut included = HashSet::new();
    let mut stack = vec![root_key.to_string()];
    while let Some(key) = stack.pop() {
        if included.contains(&key) {
            continue;
        }
        let Some(node) = nodes.get(&key) else {
            continue;
        };
        included.insert(key.clone());
        for child in &node.children {
            stack.push(child.clone());
        }
    }
    included
}

pub fn first_fork_node(nodes: &HashMap<String, VisualNode>) -> Option<VisualNode> {
    let mut best: Option<&VisualNode> = None;
    for node in nodes.values() {
        if node.children.len() < 2 {
            continue;
        }
        if best
            .map(|current| node_depth(&node.full_string) < node_depth(&current.full_string))
            .unwrap_or(true)
        {
            best = Some(node);
        }
    }
    best.cloned()
}

pub fn find_tutor_target_key(
    snapshot: &ExpandedSnapshot,
    expansion_threshold: f32,
    scroll_root: &str,
    show_all: bool,
    target_phrase: &str,
) -> Option<String> {
    let target_trie_string = phrase_to_trie_string(target_phrase);
    let base_nodes = if show_all {
        build_tree(snapshot)
    } else {
        build_visible_tree(snapshot, expansion_threshold)
    };
    let visible_keys = subtree_keys(&base_nodes, scroll_root);
    let nodes = filter_nodes_by_key_set(&base_nodes, &visible_keys);
    let mut best: Option<String> = None;
    for node in nodes.values() {
        if !target_trie_string.starts_with(&node.full_string) {
            continue;
        }
        if best
            .as_ref()
            .map(|candidate| node.full_string.len() > candidate.len())
            .unwrap_or(true)
        {
            best = Some(node.full_string.clone());
        }
    }
    best
}

pub fn deepest_visible_node(nodes: &HashMap<String, VisualNode>) -> Option<VisualNode> {
    let mut best: Option<&VisualNode> = None;
    for node in nodes.values() {
        let depth = node_depth(&node.full_string);
        let best_depth = best
            .map(|candidate| node_depth(&candidate.full_string))
            .unwrap_or(0);
        let should_replace = best.is_none()
            || depth > best_depth
            || (depth == best_depth
                && best
                    .map(|candidate| node.full_string < candidate.full_string)
                    .unwrap_or(false));
        if should_replace {
            best = Some(node);
        }
    }
    best.cloned()
}

fn ancestor_at_depth(full_string: &str, depth: usize) -> String {
    if depth == 0 {
        ROOT_SYMBOL.to_string()
    } else {
        full_string[..depth + 1].to_string()
    }
}

fn scroll_ancestor_keys(scroll_root: &str) -> Vec<String> {
    if scroll_root == ROOT_SYMBOL.to_string() {
        Vec::new()
    } else {
        let mut keys = ancestor_keys_through_root(scroll_root);
        keys.pop();
        keys
    }
}

fn root_width_for(full_string: &str) -> f64 {
    if full_string == ROOT_SYMBOL.to_string() {
        ROOT_NODE_WIDTH
    } else {
        BOX_WIDTH
    }
}

pub fn relative_depth(full_string: &str, scroll_root: &str) -> usize {
    node_depth(full_string).saturating_sub(node_depth(scroll_root))
}

pub fn compute_scroll_layout_state(
    snapshot: &ExpandedSnapshot,
    expansion_threshold: f32,
    previous_scroll_offset: usize,
) -> ScrollLayoutState {
    let visible_tree = build_visible_tree(snapshot, expansion_threshold);
    let first_fork = first_fork_node(&visible_tree);
    if first_fork.is_none() {
        let anchor = deepest_visible_node(&visible_tree);
        let rendered = subtree_keys(&visible_tree, &ROOT_SYMBOL.to_string())
            .into_iter()
            .collect::<Vec<_>>();
        let Some(anchor) = anchor else {
            return ScrollLayoutState {
                first_fork_depth: None,
                first_fork_full_string: None,
                scroll_offset: 0,
                scroll_root: ROOT_SYMBOL.to_string(),
                scroll_ancestor_keys: Vec::new(),
                rendered_node_keys: rendered,
            };
        };
        let anchor_depth = node_depth(&anchor.full_string);
        if anchor_depth == 0 {
            return ScrollLayoutState {
                first_fork_depth: None,
                first_fork_full_string: None,
                scroll_offset: 0,
                scroll_root: ROOT_SYMBOL.to_string(),
                scroll_ancestor_keys: Vec::new(),
                rendered_node_keys: rendered,
            };
        }
        let scroll_offset = previous_scroll_offset.min(anchor_depth - 1);
        let scroll_root = ancestor_at_depth(&anchor.full_string, scroll_offset);
        let rendered_node_keys = subtree_keys(&visible_tree, &scroll_root)
            .into_iter()
            .collect::<Vec<_>>();
        return ScrollLayoutState {
            first_fork_depth: None,
            first_fork_full_string: None,
            scroll_offset,
            scroll_root: scroll_root.clone(),
            scroll_ancestor_keys: scroll_ancestor_keys(&scroll_root),
            rendered_node_keys,
        };
    }

    let first_fork = first_fork.expect("checked above");
    let fork_depth = node_depth(&first_fork.full_string);
    let unclamped_offset = (SCROLL_CENTERING_WEIGHT
        * (fork_depth as f64 - SCROLL_TARGET_X_PX / SINGLE_PARENT_NODE_WIDTH_PX)
        + SCROLL_STABILITY_WEIGHT * previous_scroll_offset as f64)
        / (SCROLL_CENTERING_WEIGHT + SCROLL_STABILITY_WEIGHT);
    let scroll_offset = if fork_depth == 0 {
        0
    } else {
        unclamped_offset
            .floor()
            .max(0.0)
            .min((fork_depth - 1) as f64) as usize
    };
    let scroll_root = ancestor_at_depth(&first_fork.full_string, scroll_offset);
    let rendered_node_keys = subtree_keys(&visible_tree, &scroll_root)
        .into_iter()
        .collect::<Vec<_>>();
    ScrollLayoutState {
        first_fork_depth: Some(fork_depth),
        first_fork_full_string: Some(first_fork.full_string),
        scroll_offset,
        scroll_root: scroll_root.clone(),
        scroll_ancestor_keys: scroll_ancestor_keys(&scroll_root),
        rendered_node_keys,
    }
}

fn layout_tree(
    nodes: &mut HashMap<String, VisualNode>,
    full_string: &str,
    y: f64,
    height: f64,
    width: f64,
) {
    let (parent_key, children, parent_x, parent_width, node_z) = {
        let node = nodes.get(full_string).expect("node exists for layout");
        let parent_key = node.parent_key.clone();
        let parent_x = parent_key
            .as_ref()
            .and_then(|key| nodes.get(key))
            .map(|parent| parent.x)
            .unwrap_or(0.0);
        let parent_width = parent_key
            .as_ref()
            .and_then(|key| nodes.get(key))
            .map(|parent| parent.width)
            .unwrap_or(0.0);
        (
            parent_key,
            node.children.clone(),
            parent_x,
            parent_width,
            node.node.z,
        )
    };

    let x = if parent_key.is_none() {
        0.0
    } else {
        parent_x + parent_width
    };
    if let Some(node) = nodes.get_mut(full_string) {
        node.x = x;
        node.y = y;
        node.width = width;
        node.height = height;
    }

    if children.is_empty() {
        return;
    }

    let child_width =
        BOX_WIDTH * (1.0 + BOX_WIDTH_CHILDREN_MULTIPLIER * (children.len() as f64).ln());
    for child_key in children {
        let child = nodes.get(&child_key).expect("child exists").clone();
        let child_bottom_z = child
            .node
            .upper_siblings_inclusive_cum_z
            .expect("Missing upper_siblings_inclusive_cum_z for child layout");
        let child_height = height * ((child.node.z - node_z).exp() as f64);
        let child_bottom = y + height * ((child_bottom_z - node_z).exp() as f64);
        let child_top = child_bottom - child_height;
        layout_tree(nodes, &child_key, child_top, child_height, child_width);
    }
}

pub fn compute_laid_out_nodes(
    snapshot: &ExpandedSnapshot,
    expansion_threshold: f32,
    scroll_root: &str,
    show_all: bool,
    viewport_height: f64,
) -> Option<HashMap<String, VisualNode>> {
    if viewport_height <= 0.0 {
        return None;
    }
    let visible_tree = build_visible_tree(snapshot, expansion_threshold);
    let base_nodes = if show_all {
        build_tree(snapshot)
    } else {
        visible_tree
    };
    let visible_keys = subtree_keys(&base_nodes, scroll_root);
    let mut nodes = filter_nodes_by_key_set(&base_nodes, &visible_keys);
    if !nodes.contains_key(scroll_root) {
        return None;
    }
    layout_tree(
        &mut nodes,
        scroll_root,
        0.0,
        viewport_height,
        root_width_for(scroll_root),
    );
    Some(nodes)
}

pub fn closest_rendered_parent_key(key: &str, keys: &[String]) -> Option<String> {
    let mut parent = String::new();
    for candidate in keys {
        if candidate != key && key.starts_with(candidate) && candidate.len() > parent.len() {
            parent = candidate.clone();
        }
    }
    if parent.is_empty() {
        None
    } else {
        Some(parent)
    }
}

fn selection_masses(snapshot: &ExpandedSnapshot, keys: &[String]) -> Vec<f64> {
    let root_z = snapshot
        .get(&ROOT_SYMBOL.to_string())
        .map(|node| node.z as f64)
        .unwrap_or(0.0);
    let mut exp_z = HashMap::new();
    for key in keys {
        exp_z.insert(key.clone(), (snapshot[key].z as f64 - root_z).exp());
    }
    let mut residual_mass = exp_z.clone();
    for key in keys {
        if let Some(parent) = closest_rendered_parent_key(key, keys) {
            if let Some(parent_mass) = residual_mass.get_mut(&parent) {
                *parent_mass = (*parent_mass - exp_z[key]).max(0.0);
            }
        }
    }
    keys.iter().map(|key| residual_mass[key]).collect()
}

fn phase_ordering_hash(text: &str) -> u32 {
    let mut hash = 2166136261u32;
    for byte in text.bytes() {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(16777619);
    }
    hash
}

fn compare_phase_order_vectors(a: &[i32], b: &[i32]) -> std::cmp::Ordering {
    let max_len = a.len().max(b.len());
    for i in 0..max_len {
        let av = a.get(i).copied().unwrap_or(0);
        let bv = b.get(i).copied().unwrap_or(0);
        if av != bv {
            return av.cmp(&bv);
        }
    }
    std::cmp::Ordering::Equal
}

pub fn phase_ordered_keys(snapshot: &ExpandedSnapshot, keys: &[String]) -> Vec<String> {
    if keys.len() <= 1 {
        return keys.to_vec();
    }

    let root_z = snapshot
        .get(&ROOT_SYMBOL.to_string())
        .map(|node| node.z as f64)
        .unwrap_or(0.0);
    let mut children_by_parent: HashMap<String, Vec<String>> = HashMap::new();
    let mut root_keys = Vec::new();
    for key in keys {
        if let Some(parent) = closest_rendered_parent_key(key, keys) {
            children_by_parent
                .entry(parent)
                .or_default()
                .push(key.clone());
        } else {
            root_keys.push(key.clone());
        }
    }

    if root_keys.len() != 1 {
        let mut sorted = keys.to_vec();
        sorted.sort();
        return sorted;
    }

    let mut order_vectors = HashMap::from([(root_keys[0].clone(), Vec::<i32>::new())]);

    fn assign_vectors(
        parent_key: &str,
        snapshot: &ExpandedSnapshot,
        root_z: f64,
        children_by_parent: &HashMap<String, Vec<String>>,
        order_vectors: &mut HashMap<String, Vec<i32>>,
    ) {
        let mut child_keys = children_by_parent
            .get(parent_key)
            .cloned()
            .unwrap_or_default();
        child_keys.sort_by(|a, b| {
            let hash_a = phase_ordering_hash(&format!("{parent_key}\0{}\0{a}", snapshot[a].hash));
            let hash_b = phase_ordering_hash(&format!("{parent_key}\0{}\0{b}", snapshot[b].hash));
            hash_a.cmp(&hash_b).then_with(|| a.cmp(b))
        });
        let mut negative_mass = 0.0;
        let mut positive_mass = 0.0;
        let mut negative_count = 0;
        let mut positive_count = 0;
        let parent_vector = order_vectors.get(parent_key).cloned().unwrap_or_default();

        for child_key in child_keys {
            let child_mass = (snapshot[&child_key].z as f64 - root_z).exp();
            if negative_mass <= positive_mass {
                negative_count += 1;
                negative_mass += child_mass;
                let mut vector = parent_vector.clone();
                vector.push(-negative_count);
                order_vectors.insert(child_key.clone(), vector);
            } else {
                positive_count += 1;
                positive_mass += child_mass;
                let mut vector = parent_vector.clone();
                vector.push(positive_count);
                order_vectors.insert(child_key.clone(), vector);
            }
            assign_vectors(
                &child_key,
                snapshot,
                root_z,
                children_by_parent,
                order_vectors,
            );
        }
    }

    assign_vectors(
        &root_keys[0],
        snapshot,
        root_z,
        &children_by_parent,
        &mut order_vectors,
    );

    let mut sorted = keys.to_vec();
    sorted.sort_by(|a, b| {
        compare_phase_order_vectors(
            order_vectors.get(a).map(Vec::as_slice).unwrap_or(&[]),
            order_vectors.get(b).map(Vec::as_slice).unwrap_or(&[]),
        )
        .then_with(|| a.cmp(b))
    });
    sorted
}

pub fn timers_for_snapshot(
    snapshot: &ExpandedSnapshot,
    model: &LikelihoodModel,
    existing_timers: &VisibleNodeTimerMap,
    reset_all: bool,
    rendered_node_keys: &[String],
) -> VisibleNodeTimerMap {
    let mut next_timers = VisibleNodeTimerMap::new();
    let keys_in_snapshot = rendered_node_keys
        .iter()
        .filter(|key| snapshot.contains_key(*key))
        .cloned()
        .collect::<Vec<_>>();

    if !reset_all {
        for key in &keys_in_snapshot {
            if let Some(existing) = existing_timers.get(key) {
                next_timers.insert(key.clone(), existing.clone());
            }
        }

        let mut new_keys = keys_in_snapshot
            .iter()
            .filter(|key| !next_timers.contains_key(*key))
            .cloned()
            .collect::<Vec<_>>();
        new_keys.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
        for key in new_keys {
            let phase = closest_rendered_parent_key(&key, &keys_in_snapshot)
                .and_then(|parent| next_timers.get(&parent).map(|timer| timer.phase))
                .unwrap_or(0.5 * model.period);
            next_timers.insert(key, VisibleNodeTimer { phase });
        }
        return next_timers;
    }

    let sorted_keys = phase_ordered_keys(snapshot, &keys_in_snapshot);
    let weights = selection_masses(snapshot, &sorted_keys);
    let params = TimerSpacingParams::new(weights, model.stddev_delay, model.period)
        .with_max_iterations(DEFAULT_MAX_ITER);
    let initial = constant_phases(sorted_keys.len(), model.period);
    let phases = optimize(&params, &initial, DEFAULT_MAX_ITER)
        .map(|result| result.phases)
        .unwrap_or_else(|_| initial.clone());

    for (index, key) in sorted_keys.iter().enumerate() {
        let fallback = ((index as f64 + 0.5) * model.period) / sorted_keys.len() as f64;
        next_timers.insert(
            key.clone(),
            VisibleNodeTimer {
                phase: phases.get(index).copied().unwrap_or(fallback),
            },
        );
    }
    next_timers
}

fn logaddexp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    if a > b {
        a + (1.0 + (b - a).exp()).ln()
    } else {
        b + (1.0 + (a - b).exp()).ln()
    }
}

fn normal_logpdf(x: f64, mean: f64, stddev: f64) -> f64 {
    -0.5 * ((x - mean) / stddev).powi(2) - (stddev * (2.0 * std::f64::consts::PI).sqrt()).ln()
}

pub fn modulo_delay(time_seconds: f64, phase: f64, period: f64) -> f64 {
    let mut x = time_seconds - phase;
    x = ((x % period) + period) % period;
    x
}

pub fn timer_likelihood(time: f64, phase: f64, model: &LikelihoodModel) -> f64 {
    let x = modulo_delay(time, phase, model.period);
    let outlier_prob = model.outliers.ln() - model.period.ln();
    let not_outlier_prob = (1.0 - model.outliers).ln();
    let normal_modes = [-1.0, 0.0, 1.0]
        .into_iter()
        .map(|k| normal_logpdf(x, model.mu_delay + k * model.period, model.stddev_delay))
        .collect::<Vec<_>>();
    let mut sum_normal_modes = normal_modes[0];
    for value in normal_modes.into_iter().skip(1) {
        sum_normal_modes = logaddexp(sum_normal_modes, value);
    }
    logaddexp(outlier_prob, not_outlier_prob + sum_normal_modes)
}

pub fn build_likelihood_payload_nodes(
    snapshot: &ExpandedSnapshot,
    timers: &VisibleNodeTimerMap,
    time_seconds: f64,
    model: &LikelihoodModel,
    scroll_root: &str,
    scroll_ancestor_keys: &[String],
) -> HashMap<String, LikelihoodPayloadNode> {
    let mut nodes = HashMap::new();
    for (full_string, timer) in timers {
        if !snapshot.contains_key(full_string) {
            continue;
        }
        let likelihood = timer_likelihood(time_seconds, timer.phase, model);
        nodes.insert(
            full_string.clone(),
            LikelihoodPayloadNode {
                likelihood,
                phase: timer.phase,
            },
        );
        if full_string == scroll_root {
            for ancestor_key in scroll_ancestor_keys {
                if !snapshot.contains_key(ancestor_key) {
                    continue;
                }
                nodes.insert(
                    ancestor_key.clone(),
                    LikelihoodPayloadNode {
                        likelihood,
                        phase: timer.phase,
                    },
                );
            }
        }
    }
    nodes
}
