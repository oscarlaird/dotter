use crate::ROOT_HASH;
use crate::rolling_hash::{append_right, RHashMap, Hash};
use crate::symbol::{PadMode, XSymbol};

pub trait HasSymbol {
    fn symbol(&self) -> XSymbol;
}

pub struct SimpleDataWalker<'a, Data>
where
    Data: HasSymbol,
{
    nodes: &'a RHashMap<Data>,
    hash_stack: Vec<Hash>, // includes n, when n is returned
    symbol_stack: Vec<XSymbol>, // includes n, when n is returned
    data_stack: Vec<&'a Data>, // includes n, when n is returned
    unvisited_stack: Vec<(Hash, usize)>, // (hash, depth)
}

impl<'a, Data> SimpleDataWalker<'a, Data>
where
    Data: HasSymbol,
{
    pub fn new(nodes: &'a RHashMap<Data>) -> Self {
        assert!(nodes.contains_key(&ROOT_HASH), "root node must be in the map");
        Self {
            nodes,
            hash_stack: vec![],
            symbol_stack: vec![],
            data_stack: vec![],
            unvisited_stack: vec![(ROOT_HASH, 0)],
        }
    }

    pub fn hash_from_end(&self, i: usize) -> Hash {
        self.hash_stack[self.hash_stack.len() - i - 1]
    }

    pub fn symbol_from_end(&self, i: usize) -> XSymbol {
        self.symbol_stack[self.symbol_stack.len() - i - 1]
    }

    pub fn data_from_end(&self, i: usize) -> &'a Data {
        self.data_stack[self.data_stack.len() - i - 1]
    }

    // pub fn depth(&self) -> usize {
    //     // number of non-strict ancestors of the current node
    //     // note that this is +1 of the depth convention in unvisited_stack
    //     self.hash_stack.len()
    // }

    // type Item = ((Hash, XSymbol), &'a Data);

    pub fn next(&mut self) -> Option<((Hash, XSymbol), &'a Data)> {
        let (n_hash, n_depth) = self.unvisited_stack.pop()?;
        // truncate stacks
        self.hash_stack.truncate(n_depth);
        self.symbol_stack.truncate(n_depth);
        self.data_stack.truncate(n_depth);
        // push n to stacks
        let n_data = self.nodes.get(&n_hash).expect("child hash verified before push");
        let n_symbol = n_data.symbol();
        self.hash_stack.push(n_hash);
        self.symbol_stack.push(n_symbol);
        self.data_stack.push(n_data);
        // visit children before returning n
        let n_padmode = PadMode::for_xsymbol(n_symbol);
        for slot in (0..n_padmode.radix()).rev() {
            // .rev() because unvisited_stack is LIFO
            let c_symbol = n_padmode.slot_to_xsymbol(slot);
            let c_hash = append_right(n_hash, c_symbol);
            let c_depth = n_depth + 1;
            if self.nodes.contains_key(&c_hash) {
                self.unvisited_stack.push((c_hash, c_depth));
            }
        }
        Some(((n_hash, n_symbol), n_data))
    }
}

// TODO: AUDIT
// pub(crate) fn topo_sort<'a, Data>(nodes: &'a RHashMap<Data>) -> Vec<((Hash, XSymbol), &'a Data)>
// where
//     Data: HasSymbol,
// {
//     let walker = SimpleDataWalker::from(nodes);
//     walker.collect()
// }
