export interface ExpandedSnapshotNode {
	z: number;
	p: number | null;
	tp: number | null;
	tp0: number | null;
	a_tl0: number | null;
	upper_siblings_inclusive_cum_z: number | null;
	hash: number;
}

export type ExpandedSnapshot = Record<string, ExpandedSnapshotNode>;

export interface VisibleNodeTimer {
	phase: number;
}

export type VisibleNodeTimerMap = Record<string, VisibleNodeTimer>;

export interface ScrollLayoutState {
	firstForkDepth: number | null;
	firstForkFullString: string | null;
	scrollOffset: number;
	scrollRoot: string;
	scrollAncestorKeys: string[];
	renderedNodeKeys: string[];
}

export interface VisualNode {
	fullString: string;
	node: ExpandedSnapshotNode;
	symbol: string;
	parentKey: string | null;
	children: string[];
	x: number;
	y: number;
	width: number;
	height: number;
}

const BOX_WIDTH = 37;
const BOX_WIDTH_CHILDREN_MULTIPLIER = 1.0;
const ROOT_NODE_WIDTH = BOX_WIDTH * 1.5;

export const SCROLL_CENTERING_WEIGHT = 1;
export const SCROLL_STABILITY_WEIGHT = 4;
export const SINGLE_PARENT_NODE_WIDTH_PX = BOX_WIDTH;
export const SCROLL_TARGET_X_PX = 400;

function finalSymbol(fullString: string): string {
	if (fullString === '^') {
		return '^';
	}
	return fullString.at(-1) ?? '^';
}

export function phraseToTrieString(targetPhrase: string): string {
	return `^${targetPhrase.split(' ').join('_')}`;
}

function nodeDepth(fullString: string): number {
	if (fullString === '^') {
		return 0;
	}
	return Math.max(0, fullString.length - 1);
}

function snapshotNodePassesThreshold(
	node: ExpandedSnapshotNode,
	rootZ: number,
	expansionThreshold: number,
): boolean {
	return node.z - rootZ > expansionThreshold;
}

function buildTree(snapshot: ExpandedSnapshot): Record<string, VisualNode> {
	const nodes: Record<string, VisualNode> = {};
	for (const [fullString, node] of Object.entries(snapshot)) {
		nodes[fullString] = {
			fullString,
			node,
			symbol: finalSymbol(fullString),
			parentKey: fullString === '^' ? null : fullString.slice(0, -1),
			children: [],
			x: 0,
			y: 0,
			width: 0,
			height: 0,
		};
	}
	for (const node of Object.values(nodes)) {
		if (node.parentKey && nodes[node.parentKey]) {
			nodes[node.parentKey].children.push(node.fullString);
		}
	}
	for (const node of Object.values(nodes)) {
		node.children.sort((a, b) => a.localeCompare(b));
	}
	return nodes;
}

function rootZOf(snapshot: ExpandedSnapshot): number {
	return snapshot['^']?.z ?? 0;
}

function ancestorKeysThroughRoot(fullString: string): string[] {
	if (fullString === '^') {
		return ['^'];
	}
	const keys = ['^'];
	for (let depth = 1; depth <= nodeDepth(fullString); depth += 1) {
		keys.push(fullString.slice(0, depth + 1));
	}
	return keys;
}

function filterNodesByKeySet(
	nodes: Record<string, VisualNode>,
	keys: Set<string>,
): Record<string, VisualNode> {
	const filtered: Record<string, VisualNode> = {};
	for (const key of keys) {
		const node = nodes[key];
		if (!node) {
			continue;
		}
		filtered[key] = {
			...node,
			children: node.children.filter((childKey) => keys.has(childKey)),
		};
	}
	return filtered;
}

export function buildVisibleTree(
	snapshot: ExpandedSnapshot,
	expansionThreshold: number,
): Record<string, VisualNode> {
	const nodes = buildTree(snapshot);
	const rootZ = rootZOf(snapshot);
	const visibleKeys = new Set<string>(['^']);
	for (const [fullString, node] of Object.entries(snapshot)) {
		if (!snapshotNodePassesThreshold(node, rootZ, expansionThreshold)) {
			continue;
		}
		for (const ancestorKey of ancestorKeysThroughRoot(fullString)) {
			visibleKeys.add(ancestorKey);
		}
	}
	return filterNodesByKeySet(nodes, visibleKeys);
}

function subtreeKeys(nodes: Record<string, VisualNode>, rootKey: string): Set<string> {
	const included = new Set<string>();
	const stack = [rootKey];
	while (stack.length > 0) {
		const key = stack.pop();
		if (!key || included.has(key)) {
			continue;
		}
		const node = nodes[key];
		if (!node) {
			continue;
		}
		included.add(key);
		for (const childKey of node.children) {
			stack.push(childKey);
		}
	}
	return included;
}

export function findTutorTargetKey(
	snapshot: ExpandedSnapshot,
	expansionThreshold: number,
	scrollRoot: string,
	showAll: boolean,
	targetPhrase: string,
): string | null {
	const targetTrieString = phraseToTrieString(targetPhrase);
	const baseNodes = showAll ? buildTree(snapshot) : buildVisibleTree(snapshot, expansionThreshold);
	const visibleKeys = subtreeKeys(baseNodes, scrollRoot);
	const nodes = filterNodesByKeySet(baseNodes, visibleKeys);
	let best: string | null = null;
	for (const node of Object.values(nodes)) {
		if (!targetTrieString.startsWith(node.fullString)) {
			continue;
		}
		if (best === null || node.fullString.length > best.length) {
			best = node.fullString;
		}
	}
	return best;
}

export function firstForkNode(nodes: Record<string, VisualNode>): VisualNode | null {
	let best: VisualNode | null = null;
	for (const node of Object.values(nodes)) {
		if (node.children.length < 2) {
			continue;
		}
		if (!best || nodeDepth(node.fullString) < nodeDepth(best.fullString)) {
			best = node;
		}
	}
	return best;
}

export function deepestVisibleNode(nodes: Record<string, VisualNode>): VisualNode | null {
	let best: VisualNode | null = null;
	for (const node of Object.values(nodes)) {
		const depth = nodeDepth(node.fullString);
		const bestDepth = best === null ? -1 : nodeDepth(best.fullString);
		if (
			best === null ||
			depth > bestDepth ||
			(depth === bestDepth && node.fullString.localeCompare(best.fullString) < 0)
		) {
			best = node;
		}
	}
	return best;
}

function ancestorAtDepth(fullString: string, depth: number): string {
	if (depth <= 0) {
		return '^';
	}
	return fullString.slice(0, depth + 1);
}

function scrollAncestorKeys(scrollRoot: string): string[] {
	if (scrollRoot === '^') {
		return [];
	}
	return ancestorKeysThroughRoot(scrollRoot).slice(0, -1);
}

function rootWidthFor(fullString: string): number {
	return fullString === '^' ? ROOT_NODE_WIDTH : BOX_WIDTH;
}

export function relativeDepth(fullString: string, scrollRoot: string): number {
	return Math.max(0, nodeDepth(fullString) - nodeDepth(scrollRoot));
}

export function computeScrollLayoutState(
	snapshot: ExpandedSnapshot,
	expansionThreshold: number,
	previousScrollOffset: number,
): ScrollLayoutState {
	const visibleTree = buildVisibleTree(snapshot, expansionThreshold);
	const firstFork = firstForkNode(visibleTree);
	if (!firstFork) {
		const anchor = deepestVisibleNode(visibleTree);
		if (!anchor) {
			return {
				firstForkDepth: null,
				firstForkFullString: null,
				scrollOffset: 0,
				scrollRoot: '^',
				scrollAncestorKeys: [],
				renderedNodeKeys: Array.from(subtreeKeys(visibleTree, '^')),
			};
		}
		const anchorDepth = nodeDepth(anchor.fullString);
		if (anchorDepth <= 0) {
			return {
				firstForkDepth: null,
				firstForkFullString: null,
				scrollOffset: 0,
				scrollRoot: '^',
				scrollAncestorKeys: [],
				renderedNodeKeys: Array.from(subtreeKeys(visibleTree, '^')),
			};
		}
		const scrollOffset = Math.max(0, Math.min(previousScrollOffset, anchorDepth - 1));
		const scrollRoot = ancestorAtDepth(anchor.fullString, scrollOffset);
		return {
			firstForkDepth: null,
			firstForkFullString: null,
			scrollOffset,
			scrollRoot,
			scrollAncestorKeys: scrollAncestorKeys(scrollRoot),
			renderedNodeKeys: Array.from(subtreeKeys(visibleTree, scrollRoot)),
		};
	}

	const forkDepth = nodeDepth(firstFork.fullString);
	const unclampedOffset =
		(
			SCROLL_CENTERING_WEIGHT *
				(forkDepth - SCROLL_TARGET_X_PX / SINGLE_PARENT_NODE_WIDTH_PX) +
			SCROLL_STABILITY_WEIGHT * previousScrollOffset
		) /
		(SCROLL_CENTERING_WEIGHT + SCROLL_STABILITY_WEIGHT);
	const scrollOffset =
		forkDepth <= 0 ? 0 : Math.max(0, Math.min(Math.floor(unclampedOffset), forkDepth - 1));
	const scrollRoot = ancestorAtDepth(firstFork.fullString, scrollOffset);
	return {
		firstForkDepth: forkDepth,
		firstForkFullString: firstFork.fullString,
		scrollOffset,
		scrollRoot,
		scrollAncestorKeys: scrollAncestorKeys(scrollRoot),
		renderedNodeKeys: Array.from(subtreeKeys(visibleTree, scrollRoot)),
	};
}

function layoutTree(
	nodes: Record<string, VisualNode>,
	fullString: string,
	y: number,
	height: number,
	width: number,
): void {
	const node = nodes[fullString];
	node.x =
		node.parentKey === null || !nodes[node.parentKey]
			? 0
			: nodes[node.parentKey].x + nodes[node.parentKey].width;
	node.y = y;
	node.width = width;
	node.height = height;

	if (node.children.length === 0) {
		return;
	}

	const childWidth =
		BOX_WIDTH * (1 + BOX_WIDTH_CHILDREN_MULTIPLIER * Math.log(node.children.length));
	for (const childKey of node.children) {
		const child = nodes[childKey];
		const childBottomZ = child.node.upper_siblings_inclusive_cum_z;
		if (childBottomZ === null) {
			throw new Error(`Missing upper_siblings_inclusive_cum_z for ${child.fullString}`);
		}
		const childHeight = height * Math.exp(child.node.z - node.node.z);
		const childBottom = y + height * Math.exp(childBottomZ - node.node.z);
		const childTop = childBottom - childHeight;
		layoutTree(nodes, childKey, childTop, childHeight, childWidth);
	}
}

export function computeLaidOutNodes(
	snapshot: ExpandedSnapshot,
	expansionThreshold: number,
	scrollRoot: string,
	showAll: boolean,
	viewportHeight: number,
): Record<string, VisualNode> | null {
	if (viewportHeight <= 0) {
		return null;
	}
	const visibleTree = buildVisibleTree(snapshot, expansionThreshold);
	const baseNodes = showAll ? buildTree(snapshot) : visibleTree;
	const visibleKeys = subtreeKeys(baseNodes, scrollRoot);
	const nodes = filterNodesByKeySet(baseNodes, visibleKeys);
	if (!nodes[scrollRoot]) {
		return null;
	}
	layoutTree(nodes, scrollRoot, 0, viewportHeight, rootWidthFor(scrollRoot));
	return nodes;
}
