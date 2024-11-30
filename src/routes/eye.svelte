<script>
    import * as vision from "@mediapipe/tasks-vision";
    import { onMount, createEventDispatcher } from "svelte";

    const dispatch = createEventDispatcher();

    let faceLandmarker;
    let videoElement;
    let facial_expression_scores = [{score: 0.0, categoryName: "eyeBlinkLeft"}, {score: 0.0, categoryName: "eyeBlinkRight"}]
    $: blink_left_score = facial_expression_scores.find((shape) => shape.categoryName === "eyeBlinkLeft").score;
    $: blink_right_score = facial_expression_scores.find((shape) => shape.categoryName === "eyeBlinkRight").score;
    $: blink_left_perc = Math.max(0, Math.min(100, Math.round(blink_left_score * 100)));
    $: blink_right_perc = Math.max(0, Math.min(100, Math.round(blink_right_score * 100)));

    let blinking = false;
    let cooldown = false;
    let blink_threshold = 30;
    $: if (blink_left_perc > blink_threshold || blink_right_perc > blink_threshold) {
        if (!blinking) {
            // we just crossed the threshold
            if (!cooldown) {
                // we are not in cooldown (to avoid multiple triggers in quick succession)
                handleBlink();
                cooldown = true;
                setTimeout(() => {
                    cooldown = false;
                }, 300);
            }
        }
        blinking = true;
    } else {
        blinking = false;
    }

    let letter_idx = 0;
    let message = "Start typing: ";
    setInterval(() => {
        letter_idx = (letter_idx + 1) % 26;
    }, 140);
    $: letter = String.fromCharCode(65 + letter_idx);

    function handleBlink() {
        message += letter;
        dispatch("blink", {
            left: blink_left_score,
            right: blink_right_score,
        });
    }

    onMount(async () => {
        const filesetResolver = await vision.FilesetResolver.forVisionTasks();
        faceLandmarker = await vision.FaceLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
            delegate: 'GPU',
            },
            outputFaceBlendshapes: true,
            runningMode: 'VIDEO',
            numFaces: 1,
        });
        navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
            videoElement.srcObject = stream;
            videoElement.onloadeddata = () => {
                const predict = async () => {
                    const results = await faceLandmarker.detectForVideo(videoElement, performance.now());
                    if (results.faceBlendshapes.length) {
                        facial_expression_scores = results.faceBlendshapes[0].categories;
                    }
                    requestAnimationFrame(predict);
                }
                predict();
            }
        })
    })
</script>

{message}
<div class="flex flex-row">
    {#each Array.from({ length: 26 }) as _, i}
        <div class="text-4xl text-center w-16 h-16 border border-black flex items-center justify-center bg-red-100"
            class:font-bold={i === letter_idx}
        >{String.fromCharCode(65 + i)}</div>
    {/each}
</div>
<div class="progress_container relative w-64 h-32 border-2 border-black">
    <div class="top-0 absolute progress_bar bg-blue-200 h-full" style="width: 30%"></div>
    <div class="top-0 absolute progress_bar bg-blue-500 h-full" style="width: {blink_left_perc}%"></div>
</div>
<div class="progress_container relative w-64 h-32 border-2 border-black">
    <div class="top-0 absolute progress_bar bg-blue-200 h-full" style="width: 30%"></div>
    <div class="top-0 absolute progress_bar bg-blue-500 h-full" style="width: {blink_right_perc}%"></div>
</div>

<video id="webcam" autoplay bind:this={videoElement}
  class="hidden w-full h-full bg-black"
></video>