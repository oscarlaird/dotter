use crate::rolling_hash as rh;

pub(super) struct PUpdate {
    new_predictions: rh::RHashSet,
}

impl PUpdate {
    pub(super) fn new() -> Self {
        Self {
            new_predictions: rh::RHashSet::default(),
        }
    }

    pub(super) fn deref(&self) -> &rh::RHashSet {
        &self.new_predictions
    }

    pub(super) fn deref_mut(&mut self) -> &mut rh::RHashSet {
        &mut self.new_predictions
    }
}
