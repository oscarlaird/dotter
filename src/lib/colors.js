let color_to_rgb = {
    green: [0, 128, 0],
    red: [200, 0, 0],
    blue: [0, 0, 128],
    yellow: [234, 184, 0],
    purple: [128, 0, 128],
    orange: [255, 119, 26],
    tan: [171, 97, 47],
    gray: [72, 62, 55],
    white: [255, 255, 255],
    black: [0, 0, 0],
    undefined: [255, 255, 255],
}
let adjusted_colors = {
    // 'green': [34, 177, 76],
    // 'red': [255, 77, 77],
    // 'blue': [77, 121, 255],
    // 'yellow': [255, 225, 77],
    // 'purple': [153, 77, 255],
    // 'orange': [255, 153, 51],
    // 'tan': [204, 153, 102],
    // 'gray': [160, 160, 160], 
    'green': [8, 158, 53],
    'red': [188, 24, 24],
    'blue': [62, 89, 169],
    'yellow': [194, 179, 5],
    'purple': [96, 37, 173],
    'orange': [194, 110, 26],
    'tan': [175, 102, 30],
    'gray': [62, 62, 62],
    'white': [22, 22, 22],
    'black': [228, 228, 228],
    'white': [22, 22, 22]
};
// apply adjusted_colors to color_to_rgb
for (let color in adjusted_colors) {
    color_to_rgb[color] = adjusted_colors[color];
}


let letter_to_color = {
    a: "red",
    b: "blue",
    c: "purple",
    d: "tan",
    e: "gray",
    f: "yellow",
    g: "green",
    h: "orange",
    i: "purple",
    j: "red",
    k: "tan",
    l: "yellow",
    m: "gray",
    n: "blue",
    o: "orange",
    p: "purple",
    q: "yellow",
    r: "red",
    s: "green",
    t: "tan",
    u: "gray",
    v: "purple",
    w: "blue",
    x: "orange",
    y: "yellow",
    z: "green",
    " ": "black",
    "$": "red",
    ".": "red",
}

function color_from_letter(letter) {
    return color_to_rgb[letter_to_color[letter]];
}

export { color_to_rgb, letter_to_color, color_from_letter };
