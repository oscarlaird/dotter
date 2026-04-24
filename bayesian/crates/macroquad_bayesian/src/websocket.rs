use super::{XSymbol, XLUpdate}

enum WebSocketMsg {
    StartSession { username: String },
    Reset,
    LikelihoodUpdate


}

struct LikelihoodUpdate {
    period: f32,
    y: f32,
    nodes: RHashMap<LUpdateData>,
}

struct LUpdateData {
    likelihood: f32,
    phase: f32,
    symbol: XSymbol,
}