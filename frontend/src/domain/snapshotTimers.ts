import { optimizeTimerPhases } from '../wasm_pkg/bayesian';
import type { LikelihoodModel } from './likelihoodModel';
import { timerLikelihood } from './likelihoodModel';
import type { ExpandedSnapshot, VisibleNodeTimerMap } from './trieLayout';

interface LikelihoodPayloadNode {
	l: number;
	phase: number;
}

function effectiveWeights(snapshot: ExpandedSnapshot, keys: readonly string[]): number[] {
	const rootZ = snapshot['^']?.z ?? 0;
	const expZ: Record<string, number> = {};
	for (const key of keys) {
		expZ[key] = Math.exp(snapshot[key].z - rootZ);
	}
	const linearZ = { ...expZ };
	for (const key of keys) {
		const parent = closestRenderedParentKey(key, keys);
		if (parent) {
			linearZ[parent] = Math.max(0, linearZ[parent] - expZ[key]);
		}
	}
	return keys.map((key) => Math.max(0, linearZ[key]));
}

export function closestRenderedParentKey(key: string, keys: readonly string[]): string {
	let parent = '';
	for (const candidate of keys) {
		if (candidate !== key && key.startsWith(candidate) && candidate.length > parent.length) {
			parent = candidate;
		}
	}
	return parent;
}

function phaseOrderingHash(text: string): number {
	let hash = 2166136261;
	for (let i = 0; i < text.length; i += 1) {
		hash ^= text.charCodeAt(i);
		hash = Math.imul(hash, 16777619);
	}
	return hash >>> 0;
}

function comparePhaseOrderVectors(a: readonly number[], b: readonly number[]): number {
	const maxLen = Math.max(a.length, b.length);
	for (let i = 0; i < maxLen; i += 1) {
		const av = i < a.length ? a[i] : 0;
		const bv = i < b.length ? b[i] : 0;
		if (av !== bv) {
			return av - bv;
		}
	}
	return 0;
}

export function phaseOrderedKeys(snapshot: ExpandedSnapshot, keys: readonly string[]): string[] {
	if (keys.length <= 1) {
		return [...keys];
	}

	const rootZ = snapshot['^']?.z ?? 0;
	const childrenByParent: Record<string, string[]> = {};
	const rootKeys: string[] = [];
	for (const key of keys) {
		const parent = closestRenderedParentKey(key, keys);
		if (parent) {
			if (!childrenByParent[parent]) {
				childrenByParent[parent] = [];
			}
			childrenByParent[parent].push(key);
		} else {
			rootKeys.push(key);
		}
	}

	if (rootKeys.length !== 1) {
		return [...keys].sort();
	}

	const orderVectors: Record<string, number[]> = {
		[rootKeys[0]]: [],
	};

	const assignVectors = (parentKey: string): void => {
		const childKeys = [...(childrenByParent[parentKey] ?? [])].sort((a, b) => {
			const hashA = phaseOrderingHash(`${parentKey}\0${snapshot[a].hash}\0${a}`);
			const hashB = phaseOrderingHash(`${parentKey}\0${snapshot[b].hash}\0${b}`);
			return hashA - hashB || a.localeCompare(b);
		});
		let negativeMass = 0;
		let positiveMass = 0;
		let negativeCount = 0;
		let positiveCount = 0;
		const parentVector = orderVectors[parentKey] ?? [];

		for (const childKey of childKeys) {
			const childMass = Math.exp(snapshot[childKey].z - rootZ);
			if (negativeMass <= positiveMass) {
				negativeCount += 1;
				negativeMass += childMass;
				orderVectors[childKey] = [...parentVector, -negativeCount];
			} else {
				positiveCount += 1;
				positiveMass += childMass;
				orderVectors[childKey] = [...parentVector, positiveCount];
			}
			assignVectors(childKey);
		}
	};

	assignVectors(rootKeys[0]);

	return [...keys].sort(
		(a, b) =>
			comparePhaseOrderVectors(orderVectors[a] ?? [], orderVectors[b] ?? []) ||
			a.localeCompare(b),
	);
}

export function timersForSnapshot(
	snapshot: ExpandedSnapshot,
	model: LikelihoodModel,
	existingTimers: VisibleNodeTimerMap,
	resetAll: boolean,
	renderedNodeKeys: readonly string[],
): VisibleNodeTimerMap {
	const nextTimers: VisibleNodeTimerMap = {};
	const keysInSnapshot = renderedNodeKeys.filter((key) => key in snapshot);

	if (!resetAll) {
		for (const key of keysInSnapshot) {
			if (existingTimers[key]) {
				nextTimers[key] = existingTimers[key];
			}
		}

		const newKeys = keysInSnapshot
			.filter((key) => !nextTimers[key])
			.sort((a, b) => a.length - b.length || a.localeCompare(b));
		for (const key of newKeys) {
			const parent = closestRenderedParentKey(key, keysInSnapshot);
			nextTimers[key] = { phase: parent ? nextTimers[parent].phase : 0.5 * model.period };
		}
		return nextTimers;
	}

	const sortedKeys = phaseOrderedKeys(snapshot, keysInSnapshot);
	const weights = effectiveWeights(snapshot, sortedKeys);
	const phasesJson = optimizeTimerPhases(
		JSON.stringify(weights),
		model.stddev_delay,
		model.period,
	);
	const phases: number[] = JSON.parse(phasesJson);

	for (let i = 0; i < sortedKeys.length; i += 1) {
		nextTimers[sortedKeys[i]] = {
			phase: phases[i] ?? ((i + 0.5) * model.period) / sortedKeys.length,
		};
	}
	return nextTimers;
}

export function buildLikelihoodPayloadNodes(
	snapshot: ExpandedSnapshot,
	timers: VisibleNodeTimerMap,
	timeSeconds: number,
	model: LikelihoodModel,
	scrollRoot: string,
	scrollAncestorKeys: readonly string[],
): Record<string, LikelihoodPayloadNode> {
	const nodes: Record<string, LikelihoodPayloadNode> = {};
	for (const [fullString, timer] of Object.entries(timers)) {
		if (!(fullString in snapshot)) {
			continue;
		}
		const likelihood = timerLikelihood(timeSeconds, timer.phase, model);
		nodes[fullString] = { l: likelihood, phase: timer.phase };
		if (fullString === scrollRoot) {
			for (const ancestorKey of scrollAncestorKeys) {
				if (!(ancestorKey in snapshot)) {
					continue;
				}
				nodes[ancestorKey] = { l: likelihood, phase: timer.phase };
			}
		}
	}
	return nodes;
}
