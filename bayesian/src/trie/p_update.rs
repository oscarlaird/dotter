use crate::rolling_hash as rh;

pub(crate) struct PUpdate {
    new_predictions: rh::RHashSet,
}

impl PUpdate {
    pub(crate) fn new() -> Self {
        Self {
            new_predictions: rh::RHashSet::default(),
        }
    }

    pub(crate) fn deref(&self) -> &rh::RHashSet {
        &self.new_predictions
    }

    pub(crate) fn deref_mut(&mut self) -> &mut rh::RHashSet {
        &mut self.new_predictions
    }
}
