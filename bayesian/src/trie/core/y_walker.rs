use crate::symbol::Symbol;
use crate::rolling_hash::Hash;

pub(super) trait FromEnd<T> {
    fn from_end(&self, i: usize) -> &T;
}

impl<T> FromEnd<T> for [T] {
    #[inline]
    fn from_end(&self, i: usize) -> &T {
        &self[self.len() - i - 1]
    }
}

pub(super) struct YWalker {
    len: usize,
    a_hash: Vec<Hash>,
    a_final_token_hash: Vec<u64>,
    a_symbol: Vec<Symbol>,
    a_tp: Vec<f32>,
    a_tp0: Vec<f32>,
}

pub(super) struct YWalkerRow {
    hash: Hash,
    final_token_hash: u64,
    symbol: Symbol,
    tp: f32,
    tp0: f32,
}

impl YWalkerRow {
    pub(super) fn new(
        hash: Hash,
        final_token_hash: u64,
        symbol: Symbol,
        tp: f32,
        tp0: f32,
    ) -> Self {
        Self {
            hash,
            final_token_hash,
            symbol,
            tp,
            tp0,
        }
    }
}

impl YWalker {
    pub(super) fn root(root_hash: Hash) -> Self {
        Self {
            len: 1,
            a_hash: vec![root_hash],
            a_symbol: vec![Symbol::Start],
            a_tp: vec![0.0],
            a_tp0: vec![0.0],
            a_final_token_hash: vec![root_hash],
        }
    }

    pub(super) fn truncate(&mut self, len: usize) {
        self.len = len;
        self.a_hash.truncate(len);
        self.a_final_token_hash.truncate(len);
        self.a_symbol.truncate(len);
        self.a_tp.truncate(len);
        self.a_tp0.truncate(len);
    }

    pub(super) fn push(&mut self, row: YWalkerRow) {
        self.a_hash.push(row.hash);
        self.a_final_token_hash.push(row.final_token_hash);
        self.a_symbol.push(row.symbol);
        self.a_tp.push(row.tp);
        self.a_tp0.push(row.tp0);
        self.len += 1;
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn a_hash(&self) -> &[Hash] {
        &self.a_hash
    }

    pub(super) fn a_final_token_hash(&self) -> &[u64] {
        &self.a_final_token_hash
    }

    pub(super) fn a_symbol(&self) -> &[Symbol] {
        &self.a_symbol
    }

    pub(super) fn a_tp(&self) -> &[f32] {
        &self.a_tp
    }

    pub(super) fn a_tp0(&self) -> &[f32] {
        &self.a_tp0
    }
}
