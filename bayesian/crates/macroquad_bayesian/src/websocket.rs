use super::{XSymbol, XLUpdate};

// The frontend will be responsible for sending likelihood updates and calibration param updates
// the backend will not get physical data (except later, perhaps for offline analysis)
enum WebSocketMsg {
    StartSession { username: String },
    Reset,
    UpdateLikelihood(XLUpdate)
}

// supposing I want the lm to be in python, then should the backend websocket be owned by py or rs?
// should py call rs as a library, or should rs call py as a service?
// Both the backend py and the backend rs have to be stateful.
// So it is necessary to make the connection between them, that of a stateful connection or an embedded library.
// Should rs or py manage the db?
// The cleanest architecture is to run the model with rust, and then we don't have a language barrier.

// struct LikelihoodUpdate {
//     period: f32,
//     y: f32,
//     nodes: RHashMap<LUpdateData>,
// }

struct LUpdateData {
    likelihood: f32,
    phase: f32,
    symbol: XSymbol,
}
