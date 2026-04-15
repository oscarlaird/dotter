import type { MutableRefObject } from 'react';

function ensureAudioContext(
	audioContextRef: MutableRefObject<AudioContext | null>,
): AudioContext | null {
	const AudioContextCtor =
		window.AudioContext ??
		(window as typeof window & {
			webkitAudioContext?: typeof AudioContext;
		}).webkitAudioContext;
	if (!AudioContextCtor) {
		return null;
	}
	const ctx = audioContextRef.current ?? new AudioContextCtor();
	audioContextRef.current = ctx;
	void ctx.resume().catch(() => {});
	return ctx;
}

export function playTutorTone(
	audioContextRef: MutableRefObject<AudioContext | null>,
	frequencyHz: number,
	repetitions: number,
	options?: {
		type?: OscillatorType;
		peakGain?: number;
		duration?: number;
		gap?: number;
	},
): void {
	const ctx = ensureAudioContext(audioContextRef);
	if (!ctx) {
		return;
	}

	const startAt = ctx.currentTime + 0.005;
	const duration = options?.duration ?? 0.07;
	const gap = options?.gap ?? 0.05;
	const peakGain = options?.peakGain ?? 0.3;
	const oscType = options?.type ?? 'sine';

	for (let i = 0; i < repetitions; i += 1) {
		const osc = ctx.createOscillator();
		const gain = ctx.createGain();
		const t0 = startAt + i * (duration + gap);
		osc.type = oscType;
		osc.frequency.setValueAtTime(frequencyHz, t0);
		gain.gain.setValueAtTime(0.0001, t0);
		gain.gain.exponentialRampToValueAtTime(peakGain, t0 + 0.008);
		gain.gain.exponentialRampToValueAtTime(0.0001, t0 + duration);
		osc.connect(gain);
		gain.connect(ctx.destination);
		osc.start(t0);
		osc.stop(t0 + duration);
	}
}

export function playTutorOutlierTone(
	audioContextRef: MutableRefObject<AudioContext | null>,
): void {
	const ctx = ensureAudioContext(audioContextRef);
	if (!ctx) {
		return;
	}

	const startAt = ctx.currentTime + 0.005;
	const duration = 0.04;
	const gap = 0.03;
	for (const [idx, frequencyHz] of [520, 610].entries()) {
		const osc = ctx.createOscillator();
		const gain = ctx.createGain();
		const t0 = startAt + idx * (duration + gap);
		osc.type = 'square';
		osc.frequency.setValueAtTime(frequencyHz, t0);
		gain.gain.setValueAtTime(0.0001, t0);
		gain.gain.exponentialRampToValueAtTime(0.32, t0 + 0.006);
		gain.gain.exponentialRampToValueAtTime(0.0001, t0 + duration);
		osc.connect(gain);
		gain.connect(ctx.destination);
		osc.start(t0);
		osc.stop(t0 + duration);
	}
}
