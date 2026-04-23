use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use bayesian::calibration::VariationalParams;
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::json;
use tokio::runtime::Runtime;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;

#[derive(Clone, Debug)]
pub enum BackendCommand {
    StartSession { username: String },
    RequestNextPrior,
    Reset,
    LikelihoodUpdate { content_json: String },
}

#[derive(Clone, Debug)]
pub enum BackendEvent {
    Connected,
    Disconnected(String),
    SessionStarted {
        username: String,
        variational_params: VariationalParams,
    },
    PriorUpdate {
        content_json: String,
    },
    ResetAck,
    Error(String),
}

#[derive(Clone)]
pub struct BackendHandle {
    command_tx: UnboundedSender<BackendCommand>,
    event_rx: Arc<Mutex<mpsc::Receiver<BackendEvent>>>,
}

impl BackendHandle {
    pub fn new(url: &str) -> Self {
        let (command_tx, command_rx) = unbounded_channel();
        let (event_tx, event_rx) = mpsc::channel();
        let url = url.to_string();
        thread::spawn(move || {
            let runtime = Runtime::new().expect("tokio runtime for backend client");
            runtime.block_on(async move {
                if let Err(err) = backend_loop(url, command_rx, event_tx.clone()).await {
                    let _ = event_tx.send(BackendEvent::Error(err));
                }
            });
        });
        Self {
            command_tx,
            event_rx: Arc::new(Mutex::new(event_rx)),
        }
    }

    pub fn send(&self, command: BackendCommand) {
        let _ = self.command_tx.send(command);
    }

    pub fn drain_events(&self) -> Vec<BackendEvent> {
        let receiver = self.event_rx.lock().expect("backend event receiver lock");
        let mut events = Vec::new();
        while let Ok(event) = receiver.try_recv() {
            events.push(event);
        }
        events
    }
}

#[derive(Debug, Deserialize)]
struct BackendMessageContent {
    message: Option<String>,
    username: Option<String>,
    variational_params: Option<VariationalParams>,
}

#[derive(Debug, Deserialize)]
struct BackendMessageEnvelope {
    #[serde(rename = "type")]
    message_type: String,
    content_json: Option<String>,
    content: Option<BackendMessageContent>,
}

async fn backend_loop(
    url: String,
    mut command_rx: UnboundedReceiver<BackendCommand>,
    event_tx: mpsc::Sender<BackendEvent>,
) -> Result<(), String> {
    let (socket, _) = connect_async(url)
        .await
        .map_err(|err| format!("backend websocket connect failed: {err}"))?;
    let _ = event_tx.send(BackendEvent::Connected);
    let (mut write, mut read) = socket.split();

    loop {
        tokio::select! {
            Some(command) = command_rx.recv() => {
                let payload = outbound_json(command);
                write
                    .send(Message::Text(payload.into()))
                    .await
                    .map_err(|err| format!("backend websocket send failed: {err}"))?;
            }
            message = read.next() => {
                match message {
                    Some(Ok(Message::Text(text))) => {
                        match parse_backend_event(&text) {
                            Ok(event) => {
                                let _ = event_tx.send(event);
                            }
                            Err(err) => {
                                let _ = event_tx.send(BackendEvent::Error(err));
                            }
                        }
                    }
                    Some(Ok(Message::Close(frame))) => {
                        let reason = frame
                            .map(|frame| frame.reason.to_string())
                            .filter(|reason| !reason.is_empty())
                            .unwrap_or_else(|| "backend websocket closed".to_string());
                        let _ = event_tx.send(BackendEvent::Disconnected(reason));
                        return Ok(());
                    }
                    Some(Ok(_)) => {}
                    Some(Err(err)) => {
                        let _ = event_tx.send(BackendEvent::Disconnected(format!("backend websocket read failed: {err}")));
                        return Ok(());
                    }
                    None => {
                        let _ = event_tx.send(BackendEvent::Disconnected("backend websocket ended".to_string()));
                        return Ok(());
                    }
                }
            }
        }
    }
}

fn outbound_json(command: BackendCommand) -> String {
    match command {
        BackendCommand::StartSession { username } => {
            json!({"type": "start_session", "content": {"username": username}}).to_string()
        }
        BackendCommand::RequestNextPrior => json!({"type": "request_next_prior"}).to_string(),
        BackendCommand::Reset => json!({"type": "reset"}).to_string(),
        BackendCommand::LikelihoodUpdate { content_json } => {
            json!({"type": "likelihood_update", "content_json": content_json}).to_string()
        }
    }
}

fn parse_backend_event(text: &str) -> Result<BackendEvent, String> {
    let message: BackendMessageEnvelope = serde_json::from_str(text)
        .map_err(|err| format!("backend message decode failed: {err}"))?;
    match message.message_type.as_str() {
        "reset_ack" => Ok(BackendEvent::ResetAck),
        "session_started" => {
            let content = message
                .content
                .ok_or_else(|| "session_started missing content".to_string())?;
            let username = content
                .username
                .ok_or_else(|| "session_started missing username".to_string())?;
            let variational_params = content
                .variational_params
                .ok_or_else(|| "session_started missing variational_params".to_string())?;
            Ok(BackendEvent::SessionStarted {
                username,
                variational_params,
            })
        }
        "prior_update" => Ok(BackendEvent::PriorUpdate {
            content_json: message
                .content_json
                .ok_or_else(|| "prior_update missing content_json".to_string())?,
        }),
        "error" => {
            let content = message
                .content
                .ok_or_else(|| "error missing content".to_string())?;
            Ok(BackendEvent::Error(content.message.unwrap_or_else(|| {
                "backend returned an unspecified error".to_string()
            })))
        }
        "pong" => Ok(BackendEvent::Connected),
        other => Err(format!("unknown backend message type: {other}")),
    }
}
