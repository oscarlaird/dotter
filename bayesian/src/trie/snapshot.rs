use serde::{Deserialize, Serialize};

use crate::symbol::Symbol;

use super::NodeIndex;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrieSnapshotNode {
    pub symbol: Symbol,
    pub z: f64,
    pub likelihood: f64,
    pub children: Vec<(Symbol, NodeIndex)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrieSnapshot {
    pub nodes: Vec<TrieSnapshotNode>,
    pub root: NodeIndex,
}

#[derive(Clone, Debug)]
pub(crate) struct SnapshotWalker {
    pub(crate) node: Option<NodeIndex>,
    pub(crate) has_children_in_snapshot: bool,
    pub(crate) likelihood: f64,
}

impl TrieSnapshot {
    // walking
    pub(crate) fn root_walker(&self) -> SnapshotWalker {
        let root_node = &self.nodes[self.root];
        SnapshotWalker {
            node: Some(self.root),
            has_children_in_snapshot: !root_node.children.is_empty(),
            likelihood: root_node.likelihood,
        }
    }

    pub(crate) fn descend(&self, walker: &SnapshotWalker, target_symbol: Symbol) -> SnapshotWalker {
        let mut walker = walker.clone();
        let Some(node_index) = walker.node else {
            return walker;
        };
        let snapshot_node = &self.nodes[node_index];
        for (symbol, child_index) in &snapshot_node.children {
            if *symbol == target_symbol {
                let child_node = &self.nodes[*child_index];
                walker.node = Some(*child_index);
                walker.has_children_in_snapshot = !child_node.children.is_empty();
                walker.likelihood = child_node.likelihood;
                return walker;
            }
        }
        walker.node = None;
        walker.has_children_in_snapshot = false;
        // likelihood is inherited
        walker
    }
}
