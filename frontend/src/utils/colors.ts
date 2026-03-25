type RGB = [number, number, number];
type ColorName = 'green' | 'red' | 'blue' | 'yellow' | 'purple' | 'orange' | 'tan' | 'gray' | 'white' | 'black' | 'undefined';

const colorToRgb: Record<ColorName, RGB> = {
	green: [34, 177, 76],
	red: [255, 77, 77],
	blue: [77, 121, 255],
	yellow: [255, 225, 77],
	purple: [153, 77, 255],
	orange: [255, 153, 51],
	tan: [204, 153, 102],
	gray: [160, 160, 160],
	white: [255, 255, 255],
	black: [0, 0, 0],
	undefined: [255, 255, 255],
};

const letterToColor: Record<string, ColorName> = {
	a: 'red',
	b: 'blue',
	c: 'purple',
	d: 'tan',
	e: 'gray',
	f: 'yellow',
	g: 'green',
	h: 'orange',
	i: 'purple',
	j: 'red',
	k: 'tan',
	l: 'yellow',
	m: 'gray',
	n: 'blue',
	o: 'orange',
	p: 'purple',
	q: 'yellow',
	r: 'red',
	s: 'green',
	t: 'tan',
	u: 'gray',
	v: 'purple',
	w: 'blue',
	x: 'orange',
	y: 'yellow',
	z: 'green',
	' ': 'white',
	$: 'red',
	'.': 'red',
};

function colorFromLetter(letter: string): RGB {
	const colorName = letterToColor[letter] ?? 'undefined';
	return colorToRgb[colorName];
}

export { colorToRgb, letterToColor, colorFromLetter };
export type { RGB, ColorName };
