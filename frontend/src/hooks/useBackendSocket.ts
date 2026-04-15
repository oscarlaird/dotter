import { useCallback, useEffect, useRef, useState } from 'react';
import type { VariationalParams } from '../domain/likelihoodModel';

interface BackendMessage {
	type: string;
	content_json?: string;
	content?: {
		message?: string;
		username?: string;
		variational_params?: VariationalParams;
	};
}

interface UseBackendSocketOptions {
	enabled: boolean;
	autoStartUsername: string;
	onOpen?: () => void;
	onResetAck?: () => void | Promise<void>;
	onSessionStarted?: (content: {
		username: string;
		variational_params: VariationalParams;
	}) => void | Promise<void>;
	onPriorUpdate?: (contentJson: string) => void | Promise<void>;
	onErrorMessage?: (message: string) => void;
}

function websocketMustBeOpen(socket: WebSocket | null, action: string): WebSocket {
	if (!socket || socket.readyState !== WebSocket.OPEN) {
		throw new Error(`WebSocket must be connected before ${action}`);
	}
	return socket;
}

export function useBackendSocket({
	enabled,
	autoStartUsername,
	onOpen,
	onResetAck,
	onSessionStarted,
	onPriorUpdate,
	onErrorMessage,
}: UseBackendSocketOptions) {
	const socketRef = useRef<WebSocket | null>(null);
	const [wsStatus, setWsStatus] = useState('Connecting...');
	const [warning, setWarning] = useState<string | null>(null);

	useEffect(() => {
		if (!enabled) {
			return;
		}

		const ws = new WebSocket('ws://localhost:8000/ws');
		socketRef.current = ws;

		ws.addEventListener('open', () => {
			setWsStatus('Connected');
			setWarning(null);
			onOpen?.();
			const trimmed = autoStartUsername.trim();
			if (trimmed) {
				ws.send(JSON.stringify({ type: 'start_session', content: { username: trimmed } }));
			}
		});

		ws.addEventListener('close', () => {
			setWsStatus('Disconnected');
			setWarning('Backend disconnected. Local likelihood updates still apply.');
		});

		ws.addEventListener('error', () => {
			setWsStatus('Error');
			setWarning('Backend connection failed. Local likelihood updates still apply.');
		});

		ws.addEventListener('message', (event) => {
			void (async () => {
				const message = JSON.parse(event.data as string) as BackendMessage;

				if (message.type === 'reset_ack') {
					await onResetAck?.();
					return;
				}

				if (
					message.type === 'session_started' &&
					typeof message.content?.username === 'string' &&
					message.content.variational_params
				) {
					await onSessionStarted?.({
						username: message.content.username,
						variational_params: message.content.variational_params,
					});
					return;
				}

				if (message.type === 'prior_update' && typeof message.content_json === 'string') {
					await onPriorUpdate?.(message.content_json);
					return;
				}

				if (message.type === 'error' && message.content?.message) {
					onErrorMessage?.(message.content.message);
				}
			})().catch((err) => {
				onErrorMessage?.(err instanceof Error ? err.message : String(err));
			});
		});

		return () => {
			socketRef.current = null;
			ws.close();
		};
	}, [autoStartUsername, enabled, onErrorMessage, onOpen, onPriorUpdate, onResetAck, onSessionStarted]);

	const startSession = useCallback((username: string): void => {
		const trimmed = username.trim();
		if (!trimmed) {
			throw new Error('Username must be non-empty');
		}
		const socket = websocketMustBeOpen(socketRef.current, 'starting a session');
		socket.send(JSON.stringify({ type: 'start_session', content: { username: trimmed } }));
	}, []);

	const requestNextPrior = useCallback((): void => {
		const socket = websocketMustBeOpen(socketRef.current, 'requesting the next prior');
		socket.send(JSON.stringify({ type: 'request_next_prior' }));
	}, []);

	const reset = useCallback((): void => {
		const socket = websocketMustBeOpen(socketRef.current, 'resetting');
		socket.send(JSON.stringify({ type: 'reset' }));
	}, []);

	const sendLikelihoodUpdate = useCallback((contentJson: string): boolean => {
		const socket = socketRef.current;
		if (socket && socket.readyState === WebSocket.OPEN) {
			socket.send(JSON.stringify({ type: 'likelihood_update', content_json: contentJson }));
			setWarning(null);
			return true;
		}
		setWarning('Backend disconnected. Applied likelihoods locally only.');
		return false;
	}, []);

	return {
		wsStatus,
		warning,
		startSession,
		requestNextPrior,
		reset,
		sendLikelihoodUpdate,
	};
}
