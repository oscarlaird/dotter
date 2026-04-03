use crate::rolling_hash as rh;
use crate::rolling_hash::Hash;
use crate::symbol::Symbol;

pub(super) struct LUpdate {
    likelihoods: rh::RHashMap<f32>,
    cpc_form: bool, // complete prefix code form
}

impl LUpdate {
    pub(super) fn new() -> Self {
        Self {
            likelihoods: rh::RHashMap::default(),
            cpc_form: false,
        }
    }

    pub(super) fn deref(&self) -> &rh::RHashMap<f32> {
        &self.likelihoods
    }

    pub(super) fn deref_mut(&mut self) -> &mut rh::RHashMap<f32> {
        &mut self.likelihoods
    }

    pub(super) fn to_cpc_form(&mut self) {
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
        struct Entry {
            hash: Hash,
            likelihood: f32,
        }
        let mut new_entries: Vec<Entry> = Vec::new();
        let mut remove_entries: Vec<Hash> = Vec::new();
        for (hash, &likelihood) in self.likelihoods.iter() {
            let mut has_any_children = false;
            let mut non_preexisting_child_hashes: Vec<Hash> = Vec::new();
            for symbol in Symbol::ALL {
                let child_hash = rh::append_right(*hash, symbol.to_byte());
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
        struct Frame {
            hash: Hash,
            likelihood: f32,
            hit_count: u32,
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
            hash: 0,
            likelihood: 0.0,
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
                .fold((0, 0.0f32), |(hits, sum), &v| (hits + 1, sum + v));
            let new_hits = hit_count + hit_delta;
            let new_likelihood = likelihood + likelihood_delta;
            if new_hits == (l_tries.len() as u32) {
                result.likelihoods.insert(hash, new_likelihood);
                continue;
            }
            for symbol in Symbol::ALL {
                if symbol == Symbol::Start {
                    continue;
                }
                let child_hash = rh::append_right(hash, symbol.to_byte());
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

    pub(super) fn merge(&self, other: &Self) -> Self {
        Self::merge_many(&[self, other])
    }
}
