import { useEffect, useRef, useState } from 'react';
import * as vision from '@mediapipe/tasks-vision';

interface BlinkEvent {
	left: number;
	right: number;
}

interface EyeProps {
	onBlink: (event: BlinkEvent) => void;
}

interface FaceCategory {
	score: number;
	categoryName: string;
}

const BLINK_THRESHOLD = 25;
const COOLDOWN_MS = 250;

function Eye({ onBlink }: EyeProps) {
	const videoRef = useRef<HTMLVideoElement>(null);
	const [faceLandmarker, setFaceLandmarker] = useState<vision.FaceLandmarker | null>(null);
	const [expressionScores, setExpressionScores] = useState<FaceCategory[]>([
		{ score: 0, categoryName: 'eyeBlinkLeft' },
		{ score: 0, categoryName: 'eyeBlinkRight' },
	]);
	const [blinking, setBlinking] = useState(false);
	const [cooldown, setCooldown] = useState(false);

	const blinkLeftScore = expressionScores.find((s) => s.categoryName === 'eyeBlinkLeft')?.score ?? 0;
	const blinkRightScore = expressionScores.find((s) => s.categoryName === 'eyeBlinkRight')?.score ?? 0;
	const blinkLeftPerc = Math.max(0, Math.min(100, Math.round(blinkLeftScore * 100)));
	const blinkRightPerc = Math.max(0, Math.min(100, Math.round(blinkRightScore * 100)));

	useEffect(() => {
		const isBlinking = blinkLeftPerc > BLINK_THRESHOLD || blinkRightPerc > BLINK_THRESHOLD;
		const isOpen = blinkLeftPerc < BLINK_THRESHOLD && blinkRightPerc < BLINK_THRESHOLD;

		if (isBlinking && !blinking && !cooldown) {
			setBlinking(true);
			setCooldown(true);
			onBlink({ left: blinkLeftScore, right: blinkRightScore });
			setTimeout(() => setCooldown(false), COOLDOWN_MS);
		} else if (isOpen && blinking) {
			setBlinking(false);
		}
	}, [blinkLeftPerc, blinkRightPerc, blinking, cooldown, blinkLeftScore, blinkRightScore, onBlink]);

	// Initialise MediaPipe face landmarker
	useEffect(() => {
		let mounted = true;

		async function init() {
			const filesetResolver = await vision.FilesetResolver.forVisionTasks();
			const landmarker = await vision.FaceLandmarker.createFromOptions(filesetResolver, {
				baseOptions: {
					modelAssetPath: '/face_landmarker.task',
					delegate: 'GPU',
				},
				outputFaceBlendshapes: true,
				runningMode: 'VIDEO',
				numFaces: 1,
			});

			if (!mounted) return;
			setFaceLandmarker(landmarker);

			const stream = await navigator.mediaDevices.getUserMedia({ video: true });
			if (!videoRef.current || !mounted) return;

			videoRef.current.srcObject = stream;
			videoRef.current.onloadeddata = () => {
				const draw = async () => {
					if (!mounted || !videoRef.current || !landmarker) return;
					const results = await landmarker.detectForVideo(videoRef.current, performance.now());
					if (results.faceBlendshapes.length && mounted) {
						setExpressionScores(results.faceBlendshapes[0].categories);
					}
					if (videoRef.current && mounted) {
						videoRef.current.requestVideoFrameCallback(draw);
					}
				};
				if (videoRef.current) videoRef.current.requestVideoFrameCallback(draw);
			};
		}

		init().catch(console.error);

		return () => {
			mounted = false;
			if (videoRef.current?.srcObject) {
				(videoRef.current.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
			}
		};
	}, []);

	// Suppress unused-variable warning – faceLandmarker is only used to trigger the effect
	void faceLandmarker;

	return (
		<video
			ref={videoRef}
			id="webcam"
			autoPlay
			className="hidden w-full h-full bg-black"
		/>
	);
}

export default Eye;
export type { BlinkEvent, EyeProps };
