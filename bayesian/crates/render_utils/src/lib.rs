use std::collections::HashMap;

// Utilities for displaying the trie.
// - assign chunks
// - optimize spacing

use trie::logaddexp;
use trie::{core::XBayes, ROOT_HASH};
use trie::symbol::{XSymbol, START_SYMBOL, PadMode};
use trie::safe_float::{Float, into_f32};
use rolling_hash as rh;
use trie::dfs::{SimpleDataWalker, HasSymbol};
use serde::{Serialize, Serializer};

fn serialize_symbol<S>(symbol: &XSymbol, serializer: S) -> Result<S::Ok, S::Error> 
    where S: Serializer
{
    serializer.serialize_char(*symbol as char)
}

fn serialize_float<S>(float: &Float, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer
{
    serializer.serialize_f32(into_f32(*float))
}

fn serialize_optional_float<S>(float: &Option<Float>, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer
{
    match float {
        Some(float) => serializer.serialize_some(&into_f32(*float)),
        None => serializer.serialize_none(),
    }
}



#[derive(Serialize)]
pub struct VizSnapshotNodeFields<T> {
    pub value: T,
    #[serde(serialize_with = "serialize_float")]
    pub z: Float,
    #[serde(serialize_with = "serialize_symbol")]
    pub symbol: XSymbol,
    #[serde(serialize_with = "serialize_optional_float")]
    pub upper_siblings_inclusive_cum_z: Option<Float>,
    // debug
    #[serde(serialize_with = "serialize_optional_float")]
    pub p: Option<Float>,
    #[serde(serialize_with = "serialize_optional_float")]
    pub tp: Option<Float>,
    #[serde(serialize_with = "serialize_optional_float")]
    pub tp0: Option<Float>,
    #[serde(serialize_with = "serialize_optional_float")]
    pub a_tl0: Option<Float>,
}

pub type VizSnapshotNodeStringField = VizSnapshotNodeFields<String>;
pub type VizSnapshotNodeHashField = VizSnapshotNodeFields<rh::Hash>;

pub type HashSymbolPair = (rh::Hash, XSymbol);

struct Chunk {
    length: usize,
    sum_exclusive_z: Float,
}

struct SetChunksData {
    symbol: XSymbol,
    chunk_start: rh::Hash,
}

pub struct SetChunksResult {
    chunks: rh::RHashMap<Chunk>,
    node_to_chunk_start: rh::RHashMap<SetChunksData>,
}

pub trait HasExclusiveZ {
    fn exclusive_z(&self) -> Float;
}

pub fn set_chunks<T>(
    exclusive_z_data: &rh::RHashMap<T>
) -> SetChunksResult
where
    T: HasExclusiveZ,
    T: HasSymbol
{
    let chunk_threshold = Float::from(0.002f32);
    let mut chunks = rh::RHashMap::<Chunk>::default();
    let mut result_data = rh::RHashMap::<SetChunksData>::default();
    let mut data_walker = SimpleDataWalker::new(&exclusive_z_data);
    while let Some(((n_hash, n_symbol), n_data)) = data_walker.next() {
        // let same_chunk = data_walker.data_stack.len() > 1 && data_walker.data_from_end(1).exclusive_z() < chunk_threshold;
        let same_chunk = n_hash != ROOT_HASH && data_walker.data_from_end(1).exclusive_z() < chunk_threshold;
        if same_chunk {
            let p_hash = data_walker.hash_from_end(1);
            let p_result = result_data.get(&p_hash).unwrap();
            let p_chunk_start = p_result.chunk_start;
            let chunk = chunks.get_mut(&p_chunk_start).unwrap();
            chunk.length += 1;
            chunk.sum_exclusive_z = logaddexp(chunk.sum_exclusive_z, n_data.exclusive_z());
            let n_chunk_start = p_chunk_start;
            result_data.insert(n_hash, SetChunksData { chunk_start: n_chunk_start, symbol: n_symbol });
        } else { // !same_chunk
            let new_chunk = Chunk {
                length: 1,
                sum_exclusive_z: n_data.exclusive_z(),
            };
            chunks.insert(n_hash, new_chunk);
            result_data.insert(n_hash, SetChunksData { chunk_start: n_hash, symbol: n_symbol });
        }
    }
    SetChunksResult {
        chunks,
        node_to_chunk_start: result_data,
    }
}


pub fn snapshot_by_string(
    // returns a map of nodes, keyed by string (for external consumption)
    trie: &XBayes,
    nodes_list: Vec<(rh::Hash, XSymbol)>
) -> HashMap<String, VizSnapshotNodeHashField> {
    let nodes_list_hashes = nodes_list.iter().map(|(hash, _)| *hash).collect::<Vec<_>>();
    let snapshot_by_hash = snapshot_fields_for_node_list(trie, nodes_list_hashes);
    // swap out the string for the hash as the primary key
    let snapshot_by_string = snapshot_by_hash.into_iter()
        .map(|(hash, n_string)| (n_string.value,
            VizSnapshotNodeHashField {
                value: hash,
                z: n_string.z,
                p: n_string.p,
                tp: n_string.tp,
                tp0: n_string.tp0,
                a_tl0: n_string.a_tl0,
                symbol: n_string.symbol,
                upper_siblings_inclusive_cum_z: n_string.upper_siblings_inclusive_cum_z,
            }))
        .collect::<HashMap<String, VizSnapshotNodeHashField>>();
    snapshot_by_string
}

pub fn snapshot_fields_for_node_list(
        trie: &XBayes,
        nodes_list_hashes: Vec<rh::Hash>
    ) -> rh::RHashMap<VizSnapshotNodeStringField> {
    // add all siblings to that the frontend knows how to space things
    let nodes_set = nodes_list_hashes.into_iter().collect::<rh::RHashSet>();
    let mut snapshot_by_hash: rh::RHashMap<VizSnapshotNodeStringField> = rh::RHashMap::default();
    snapshot_by_hash.insert(trie::ROOT_HASH, VizSnapshotNodeStringField {
        value: String::from(trie::ROOT_STRING),
        z: trie.nodes.get(&trie::ROOT_HASH).unwrap().if_root_then_z,
        p: None,
        tp: None,
        tp0: None,
        a_tl0: None,
        symbol: trie::symbol::START_SYMBOL,
        upper_siblings_inclusive_cum_z: None,
    });

    struct Frame {
        n_hash: rh::Hash,
        n_symbol: XSymbol,
        n_depth: u16,
    }
    // a frame is a node that *may* have children that should be added to the snapshot
    // by the time we see a frame, it should already be in the snapshot
    // the frame list is a visit list
    let mut full_string = String::new(); // character stack
    let mut frames: Vec<Frame> = vec![Frame { n_hash: trie::ROOT_HASH, n_symbol: START_SYMBOL, n_depth: 0 }];
    while let Some(Frame { n_hash, n_symbol, n_depth }) = frames.pop() {
        full_string.truncate(n_depth as usize);
        full_string.push(n_symbol as char);
        debug_assert!(snapshot_by_hash.contains_key(&n_hash), "frame node not in snapshot");
        let n_padmode = PadMode::for_xsymbol(n_symbol);
        let c_present = (0..n_padmode.radix()).map(|slot| {
            let c_byte = n_padmode.slot_to_xsymbol(slot);
            let c_hash = rh::append_right(n_hash, c_byte);
            nodes_set.contains(&c_hash)
        }).collect::<Vec<_>>();
        let is_interior = c_present.iter().any(|&p| p);
        if !is_interior {
            continue;
        }
        debug_assert!(trie.nodes.contains_key(&n_hash), "frame node not in trie");
        let n_node = trie.nodes.get(&n_hash).unwrap();
        let c_z = n_node.c_z;
        // Compute the partial sums of c_z using logaddexp, i.e., log-sum-exp over upper siblings
        let mut c_upper_siblings_inclusive_cum_z = vec![Float::NEG_INFINITY; n_padmode.radix()].into_boxed_slice();
        let mut accum = Float::NEG_INFINITY;
        for i in 0..n_padmode.radix() {
            accum = trie::logaddexp(accum, c_z[i]);
            c_upper_siblings_inclusive_cum_z[i] = accum;
        }
    
        for slot in 0..n_padmode.radix() {
            if !c_present[slot] {
                continue;
            }
            let c_symbol = n_padmode.slot_to_xsymbol(slot);
            let c_byte = c_symbol;
            let c_hash = rh::append_right(n_hash, c_byte);
            let mut c_string = full_string.clone();
            c_string.push(c_symbol as char);
            let (z, p, tp, tp0, a_tl0) = n_node.edge_snapshot_fields(slot);
            snapshot_by_hash.insert(c_hash, VizSnapshotNodeStringField {
                value: c_string,
                z,
                p: Some(p),
                tp: Some(tp),
                tp0: Some(tp0),
                a_tl0: Some(a_tl0),
                symbol: c_symbol,
                upper_siblings_inclusive_cum_z: Some(c_upper_siblings_inclusive_cum_z[slot]),
            });
            frames.push(Frame {
                n_hash: c_hash,
                n_symbol: c_symbol,
                n_depth: n_depth + 1,
            })
        }
    }
    snapshot_by_hash
}