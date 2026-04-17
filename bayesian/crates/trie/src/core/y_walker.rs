use crate::symbol::{XSymbol, START_SYMBOL};
use crate::rolling_hash::Hash;
use crate::safe_float::{Float, ZERO};
use crate::bpe::TokenLexIndex;

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
    a_final_token_lexindex: Vec<TokenLexIndex>,
    a_symbol: Vec<XSymbol>,
    a_slot: Vec<usize>,
    a_tp: Vec<Float>,
    a_tp0: Vec<Float>,
}

pub(super) struct YWalkerRow {
    hash: Hash,
    final_token_lexindex: TokenLexIndex,
    symbol: XSymbol,
    slot: usize,
    tp: Float,
    tp0: Float,
}

impl YWalkerRow {
    pub(super) fn new(
        hash: Hash,
        final_token_lexindex: TokenLexIndex,
        symbol: XSymbol,
        slot: usize,
        tp: Float,
        tp0: Float,
    ) -> Self {
        Self {
            hash,
            final_token_lexindex,
            symbol,
            slot,
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
            a_symbol: vec![START_SYMBOL],
            a_slot: vec![usize::MAX],
            a_tp: vec![ZERO],
            a_tp0: vec![ZERO],
            a_final_token_lexindex: vec![TokenLexIndex::INVALID],
        }
    }

    pub(super) fn truncate(&mut self, len: usize) {
        self.len = len;
        self.a_hash.truncate(len);
        self.a_final_token_lexindex.truncate(len);
        self.a_symbol.truncate(len);
        self.a_slot.truncate(len);
        self.a_tp.truncate(len);
        self.a_tp0.truncate(len);
    }

    pub(super) fn push(&mut self, row: YWalkerRow) {
        self.a_hash.push(row.hash);
        self.a_final_token_lexindex.push(row.final_token_lexindex);
        self.a_symbol.push(row.symbol);
        self.a_slot.push(row.slot);
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

    pub(super) fn a_final_token_lexindex(&self) -> &[TokenLexIndex] {
        &self.a_final_token_lexindex
    }

    pub(super) fn a_symbol(&self) -> &[XSymbol] {
        &self.a_symbol
    }

    pub(super) fn a_slot(&self) -> &[usize] {
        &self.a_slot
    }

    pub(super) fn a_tp(&self) -> &[Float] {
        &self.a_tp
    }

    pub(super) fn a_tp0(&self) -> &[Float] {
        &self.a_tp0
    }
}
