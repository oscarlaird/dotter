import type { DelayPair } from './stats';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface PossibleAncestor {
	ancestor: string;
	suffix: string;
	bounds: [number, number];
}

export interface TrieNode {
	val: string;
	letter: string;
	delay_pairs: DelayPair[];
	is_visible: boolean;
	likelihood: number;
	unpushed_likelihood: number;
	unpushed_delay_pairs: DelayPair[];
	children: TrieNode[];
	has_children: boolean;
	prior: number;
	post_Z: number;
	included_ancestors: Set<string>;
	possible_ancestors: PossibleAncestor[];
	initialized?: boolean;
	// Visualiser-only fields (added at render time)
	location?: { x: number; y: number };
	size_height?: number;
	size_width?: number;
	phase?: number;
	go_live_time?: number;
}

export interface LmEntry {
	cum: number[];
	stop_prob: number;
	prior_ill: number;
}

export type LmMap = Record<string, LmEntry>;

export type PrefixRangePrecomp = Record<string, [number, number]>;

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

export function logaddexp(a: number, b: number): number {
	if (a === -Infinity) return b;
	if (b === -Infinity) return a;
	if (a > b) return a + Math.log(1 + Math.exp(b - a));
	return b + Math.log(1 + Math.exp(a - b));
}

function logsubexp(a: number, b: number): number {
	if (b > a) throw new Error(`logsubexp: b > a  (a=${a}, b=${b})`);
	return a + Math.log(1 - Math.exp(b - a));
}

export function F(lmResult: LmEntry, bounds: [number, number]): number {
	const [start, end] = bounds;
	if (start === 0) return Math.log(lmResult.cum[end - 1]);
	return Math.log(lmResult.cum[end - 1] - lmResult.cum[start - 1]);
}

// ---------------------------------------------------------------------------
// Trie construction
// ---------------------------------------------------------------------------

const MAX_TOKEN_LENGTH = 16;

function getPossibleAncestors(val: string, prefixRangePrecomp: PrefixRangePrecomp): PossibleAncestor[] {
	const ancestors: PossibleAncestor[] = [];
	for (let i = Math.max(0, val.length - MAX_TOKEN_LENGTH); i < val.length; i++) {
		const ancestor = val.slice(0, i);
		const suffix = val.slice(i);
		if (suffix in prefixRangePrecomp) {
			ancestors.push({ ancestor, suffix, bounds: prefixRangePrecomp[suffix] });
		}
	}
	return ancestors;
}

export const root_node: TrieNode = {
	val: '',
	letter: '',
	delay_pairs: [],
	is_visible: true,
	likelihood: 0,
	unpushed_likelihood: 0,
	unpushed_delay_pairs: [],
	children: [],
	has_children: false,
	prior: 0,
	post_Z: 0,
	included_ancestors: new Set(),
	possible_ancestors: [],
};

function initializeChildren(node: TrieNode, prefixRangePrecomp: PrefixRangePrecomp): void {
	node.unpushed_likelihood = 0;
	node.unpushed_delay_pairs = [];
	const children: TrieNode[] = [];
	for (const c of 'abcdefghijklmnopqrstuvwxyz $') {
		const childVal = node.val + c;
		children.push({
			val: childVal,
			letter: c,
			delay_pairs: [...node.delay_pairs],
			is_visible: false,
			likelihood: node.likelihood,
			unpushed_likelihood: 0,
			unpushed_delay_pairs: [],
			children: [],
			has_children: false,
			prior: -Infinity,
			post_Z: -Infinity,
			included_ancestors: new Set(),
			possible_ancestors: getPossibleAncestors(childVal, prefixRangePrecomp),
			initialized: false,
		});
	}
	node.children = children;
	node.has_children = true;
}

// ---------------------------------------------------------------------------
// Core algorithm
// ---------------------------------------------------------------------------

export function pushl_recalc_post_Z_new(
	node: TrieNode,
	newLikelihoods: Record<string, { likelihood: number; delay_pair: DelayPair }>,
): void {
	const { likelihood: timerLikelihood, delay_pair: delayPair } = newLikelihoods[node.val];
	const hasVisibleChildren = node.children.some((child) => child.is_visible);

	if (!hasVisibleChildren) {
		node.likelihood += timerLikelihood;
		node.delay_pairs.push(delayPair);
		node.post_Z += timerLikelihood;
		node.unpushed_likelihood += timerLikelihood;
		node.unpushed_delay_pairs.push(delayPair);
		return;
	}

	let post_Z = -Infinity;
	for (const child of node.children) {
		if (child.val in newLikelihoods) {
			pushl_recalc_post_Z_new(child, newLikelihoods);
		} else {
			child.likelihood += timerLikelihood;
			child.delay_pairs.push(delayPair);
			child.post_Z += timerLikelihood;
			child.unpushed_likelihood += timerLikelihood;
			child.unpushed_delay_pairs.push(delayPair);
		}
		post_Z = logaddexp(post_Z, child.post_Z);
	}
	node.likelihood += timerLikelihood;
	node.delay_pairs.push(delayPair);
	node.post_Z = post_Z;
}

export function set_viztrie_new(
	node: TrieNode,
	lm: LmMap,
	threshold: number,
	prefixRangePrecomp: PrefixRangePrecomp,
	pDATA: number,
): void {
	if (!node.initialized) {
		for (const possibleAncestor of node.possible_ancestors) {
			if (
				!node.included_ancestors.has(possibleAncestor.ancestor) &&
				possibleAncestor.ancestor in lm
			) {
				node.prior = logaddexp(
					node.prior,
					lm[possibleAncestor.ancestor].prior_ill + F(lm[possibleAncestor.ancestor], possibleAncestor.bounds),
				);
				node.included_ancestors.add(possibleAncestor.ancestor);
			}
		}
		if (node.letter === '$') {
			const parentVal = node.val.slice(0, -1);
			if (parentVal in lm) {
				node.prior = lm[parentVal].prior_ill + lm[parentVal].stop_prob;
			}
		}
		node.post_Z = node.prior + node.likelihood;
		node.initialized = true;
	}

	node.is_visible = node.post_Z - pDATA > threshold;
	if (!node.is_visible) return;

	if (node.has_children) {
		for (const child of node.children) {
			child.likelihood += node.unpushed_likelihood;
			child.delay_pairs.push(...node.unpushed_delay_pairs);
			child.post_Z += node.unpushed_likelihood;
			child.unpushed_likelihood += node.unpushed_likelihood;
			child.unpushed_delay_pairs.push(...node.unpushed_delay_pairs);
		}
		node.unpushed_likelihood = 0;
		node.unpushed_delay_pairs = [];
	} else {
		initializeChildren(node, prefixRangePrecomp);
	}

	for (const child of node.children) {
		set_viztrie_new(child, lm, threshold, prefixRangePrecomp, pDATA);
	}
}

function updatePriorNew(
	updateRootVal: string,
	node: TrieNode,
	lm: LmMap,
	prefixRangePrecomp: PrefixRangePrecomp,
): void {
	const suffix = node.val.slice(updateRootVal.length);
	let updatePriorContribution: number;

	if (suffix === '$') {
		updatePriorContribution = lm[updateRootVal].prior_ill + lm[updateRootVal].stop_prob;
	} else if (suffix === '') {
		updatePriorContribution = -Infinity;
	} else {
		const bounds = prefixRangePrecomp[suffix];
		if (!bounds) throw new Error(`No bounds found for suffix "${suffix}"`);
		updatePriorContribution = lm[updateRootVal].prior_ill + F(lm[updateRootVal], bounds);
	}

	const oldPrior = node.prior;
	node.prior = logaddexp(node.prior, updatePriorContribution);

	if (oldPrior === -Infinity) {
		node.post_Z = node.prior + node.likelihood;
	} else {
		node.post_Z += node.prior - oldPrior;
	}

	const hasChildrenAffected = node.children.some(
		(child) => suffix + child.letter in prefixRangePrecomp,
	);
	const validPostZ = !hasChildrenAffected || !node.has_children;
	if (validPostZ) return;

	let post_Z = -Infinity;
	for (const child of node.children) {
		child.likelihood += node.unpushed_likelihood;
		child.delay_pairs.push(...node.unpushed_delay_pairs);
		child.post_Z += node.unpushed_likelihood;
		child.unpushed_likelihood += node.unpushed_likelihood;
		child.unpushed_delay_pairs.push(...node.unpushed_delay_pairs);

		if (suffix + child.letter in prefixRangePrecomp || suffix + child.letter === '$') {
			updatePriorNew(updateRootVal, child, lm, prefixRangePrecomp);
		}
		post_Z = logaddexp(post_Z, child.post_Z);
	}
	node.unpushed_likelihood = 0;
	node.unpushed_delay_pairs = [];
	node.post_Z = post_Z;
}

function getNodeByVal(node: TrieNode, val: string): TrieNode | null {
	if (node.val === val) return node;
	const nextLetter = val[node.val.length];
	const child = node.children.find((c) => c.letter === nextLetter);
	if (!child) return null;
	return getNodeByVal(child, val);
}

function grabPostZNew(node: TrieNode, targetVal: string, targetNodeOldPostZ: number): number {
	if (node.val === targetVal) return targetNodeOldPostZ;

	const nextLetter = targetVal[node.val.length];
	const child = node.children.find((c) => c.letter === nextLetter);
	if (!child) {
		throw new Error(
			`No child found for letter "${nextLetter}" in node "${node.val}" when target_val="${targetVal}"`,
		);
	}

	const selfOldPostZ = node.post_Z;
	const oldChildPostZ = grabPostZNew(child, targetVal, targetNodeOldPostZ);
	node.post_Z = logsubexp(node.post_Z, oldChildPostZ);
	node.post_Z = logaddexp(node.post_Z, child.post_Z);

	if (isNaN(node.post_Z)) {
		throw new Error(
			`Invalid post_Z for node "${node.val}": oldChildPostZ=${oldChildPostZ}, child.post_Z=${child.post_Z}`,
		);
	}
	return selfOldPostZ;
}

export function update_prior_pipeline(
	trie: TrieNode,
	val: string,
	lm: LmMap,
	prefixRangePrecomp: PrefixRangePrecomp,
): void {
	const node = getNodeByVal(trie, val);
	if (node === null) {
		console.warn('update_prior_pipeline: node is null for val', val);
		return;
	}
	const targetNodeOldPostZ = node.post_Z;
	updatePriorNew(val, node, lm, prefixRangePrecomp);
	console.log('Before grab, node.post_Z', node.post_Z, 'for val', val, 'trie post_Z', trie.post_Z);
	grabPostZNew(trie, val, targetNodeOldPostZ);
	console.log('After grab, node.post_Z', node.post_Z, 'for val', val, 'trie post_Z', trie.post_Z);

	if (node.post_Z > trie.post_Z) {
		console.warn(
			'update_prior_pipeline: node.post_Z > trie.post_Z for val',
			val,
			'node.post_Z',
			node.post_Z,
			'trie.post_Z',
			trie.post_Z,
		);
		throw new Error('update_prior_pipeline: node.post_Z > trie.post_Z');
	}
}

export function get_best_descendant(node: TrieNode): TrieNode {
	if (!node.has_children || node.letter === '$') return node;
	let bestChild = node.children[0];
	for (const child of node.children) {
		if (child.post_Z > bestChild.post_Z) bestChild = child;
	}
	return get_best_descendant(bestChild);
}
