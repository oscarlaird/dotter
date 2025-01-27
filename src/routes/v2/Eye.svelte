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
    let cooldown_time = 250;
    let blink_threshold = 25;
    
    $: if ((blink_left_perc > blink_threshold || blink_right_perc > blink_threshold) && !blinking && !cooldown) {
        // Eyes just closed and we're not in cooldown
        blinking = true;
        cooldown = true;
        handleBlink();
        setTimeout(() => {
            cooldown = false;
        }, cooldown_time);
    } else if (blink_left_perc < blink_threshold && blink_right_perc < blink_threshold && blinking) {
        blinking = false;
    }

    function handleBlink() {
        dispatch("blink", {
            left: blink_left_score,
            right: blink_right_score,
        });
    }

    onMount(async () => {
        const filesetResolver = await vision.FilesetResolver.forVisionTasks();
        faceLandmarker = await vision.FaceLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
            // modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
            modelAssetPath: '/face_landmarker.task',
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
        setInterval(() => {
            console.log({blink_left_perc, blink_right_perc, blinking, cooldown});
        }, 1000);
    })
</script>

<video id="webcam" autoplay bind:this={videoElement}
  class="hidden w-full h-full bg-black"
></video>