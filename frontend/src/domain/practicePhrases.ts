import practicePhrasesText from '../pages/practice-phrases.txt?raw';

const N_SKIP_PRACTICE_PHRASES = 6;

function formatPracticePhrase(phrase: string): string {
	return ` ${phrase}Z`;
}

export const PRACTICE_PHRASES = practicePhrasesText
	.split('\n')
	.map((line) => line.trim())
	.filter((line) => line.length > 0)
	.map(formatPracticePhrase);

export function randomPracticePhrase(excluding?: string): string {
	const eligiblePhrases = PRACTICE_PHRASES.slice(N_SKIP_PRACTICE_PHRASES);
	const source = eligiblePhrases.length > 0 ? eligiblePhrases : PRACTICE_PHRASES;
	const candidates =
		source.length > 1 && excluding ? source.filter((phrase) => phrase !== excluding) : source;
	return candidates[Math.floor(Math.random() * candidates.length)];
}
