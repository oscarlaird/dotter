// import bisect
// def get_prefix_range(prefix, tokens):
//     next_prefix = prefix[:-1] + chr(ord(prefix[-1]) + 1)
//     return bisect.bisect_left(tokens, prefix), bisect.bisect_right(tokens, next_prefix)

// dummy_tokens = ['ochastic', 'oci', 'ocia', 'ocity', 'ock', 'ocker', 'ocket', 'ockey', 'oco', 'ocoa']
// get_prefix_range('ock', dummy_tokens)
// # >>> (4, 8)
// get_prefix_range('z', dummy_tokens)
// # >>> (10, 10)
// get_prefix_range('a', dummy_tokens)
// # >>> (0, 0)
// get_prefix_range('oci', dummy_tokens)
// # >>> (1, 4)
// get_prefix_range('oco', dummy_tokens)
// # >>> (8, 10)
// get_prefix_range('oc', dummy_tokens)
// # >>> (0, 10)

// JavaScript implementation
// Binary search implementation
function binarySearch(target, tokens) {
    let left = 0;
    let right = tokens.length;
    
    while (left < right) {
        const mid = Math.floor((left + right) / 2);
        if (tokens[mid] < target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

function getPrefixRange(prefix, tokens) {
    // Get next prefix by incrementing last character
    const nextPrefix = prefix.slice(0, -1) + 
        String.fromCharCode(prefix.charCodeAt(prefix.length - 1) + 1);
    
    // Find start and end indices using binary search
    const start = binarySearch(prefix, tokens);
    const end = binarySearch(nextPrefix, tokens);
    
    return [start, end];
}

describe('getPrefixRange', () => {
    const dummyTokens = ['ochastic', 'oci', 'ocia', 'ocity', 'ock', 'ocker', 'ocket', 'ockey', 'oco', 'ocoa'];

    test('finds range for prefix in middle of array', () => { expect(getPrefixRange('ock', dummyTokens)).toEqual([4, 8]); });
    test('handles prefix after all tokens', () => { expect(getPrefixRange('z', dummyTokens)).toEqual([10, 10]); });
    test('handles prefix before all tokens', () => { expect(getPrefixRange('a', dummyTokens)).toEqual([0, 0]); });
    test('finds range for prefix with multiple matches', () => { expect(getPrefixRange('oci', dummyTokens)).toEqual([1, 4]); });
    test('finds range at end of array', () => { expect(getPrefixRange('oco', dummyTokens)).toEqual([8, 10]); }); 
    test('finds range spanning whole array', () => { expect(getPrefixRange('oc', dummyTokens)).toEqual([0, 10]); }); 
});


// def lm_F(prefix, tokens, probs_cumulative):
//     prefix_range = get_prefix_range(prefix, tokens)
//     print(prefix_range)
//     if prefix_range[0] <= 0 and prefix_range[1] <= 0:
//         return 0.0
//     if prefix_range[0] <= 0:
//         return probs_cumulative[prefix_range[1]-1]
//     if prefix_range[1] <= 0:
//         return 0.0
//     return probs_cumulative[prefix_range[1]-1] - probs_cumulative[prefix_range[0]-1]

// dummy_tokens = ['ochastic', 'oci', 'ocia', 'ocity', 'ock', 'ocker', 'ocket', 'ockey', 'oco', 'ocoa']
// dummy_probs = np.array([0, 1, 2, 3, 4, 0, 6, 7, 8, 9], dtype=np.float32)
// dummy_probs /= dummy_probs.sum()
// dummy_probs_cumulative = np.cumsum(dummy_probs)
// dummy_probs_cumulative
// print(list(zip(dummy_tokens, dummy_probs_cumulative)))
// lm_F("z", dummy_tokens, dummy_probs_cumulative)
// # >>> 0.0
// lm_F("a", dummy_tokens, dummy_probs_cumulative)
// # >>> 0.0
// lm_F("oci", dummy_tokens, dummy_probs_cumulative)
// # >>> 0.15 (6/40)
// lm_F("oco", dummy_tokens, dummy_probs_cumulative)
// # >>> 0.425 (8+9=17/40)
// lm_F("oc", dummy_tokens, dummy_probs_cumulative)
// # >>> 1.0 
// lm_F("ocity", dummy_tokens, dummy_probs_cumulative)
// # >>> 0.075 (3/40)
function lmF(prefix, tokens, probsCumulative) {
    const prefixRange = getPrefixRange(prefix, tokens);
    
    if (prefixRange[0] <= 0 && prefixRange[1] <= 0) {
        return 0.0;
    }
    if (prefixRange[0] <= 0) {
        return probsCumulative[prefixRange[1]-1];
    }
    if (prefixRange[1] <= 0) {
        return 0.0;
    }
    return probsCumulative[prefixRange[1]-1] - probsCumulative[prefixRange[0]-1];
}

describe('lmF', () => {
    const dummyTokens = ['ochastic', 'oci', 'ocia', 'ocity', 'ock', 'ocker', 'ocket', 'ockey', 'oco', 'ocoa'];
    const dummyProbs = [0, 1, 2, 3, 4, 0, 6, 7, 8, 9];
    const probsSum = dummyProbs.reduce((a, b) => a + b, 0);
    const dummyProbsCumulative = dummyProbs.map((sum => value => sum += value/probsSum)(0));

    test('handles prefix after all tokens', () => { expect(lmF('z', dummyTokens, dummyProbsCumulative)).toBe(0.0); });
    test('handles prefix before all tokens', () => { expect(lmF('a', dummyTokens, dummyProbsCumulative)).toBe(0.0); });
    test('finds probability for prefix with multiple matches', () => { expect(lmF('oci', dummyTokens, dummyProbsCumulative)).toBeCloseTo(0.15); });
    test('finds probability for prefix at end of array', () => { expect(lmF('oco', dummyTokens, dummyProbsCumulative)).toBeCloseTo(0.425); });
    test('finds probability spanning whole array', () => { expect(lmF('oc', dummyTokens, dummyProbsCumulative)).toBeCloseTo(1.0); });
    test('finds probability for exact token match', () => { expect(lmF('ocity', dummyTokens, dummyProbsCumulative)).toBeCloseTo(0.075); });
});

export { lmF };