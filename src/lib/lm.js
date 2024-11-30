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

export { binarySearch, getPrefixRange, lmF };