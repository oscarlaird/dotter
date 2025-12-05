<script>
    import { onMount } from 'svelte';
    import Eye from '../Eye.svelte';
    let canvas_element;
    let ctx;
    let fps = 0;
    let lastFrameTime = 0;
    let pause_timer = false;
    let wpm_timer_start = 0;
    let time_elapsed = 0;
    let blink_to_click = true;
    let play_click_sound = false;
    let developer_visualizer = false;
    let show_boxes = true;
    let use_visual_tutor = false;
    const FPS_SMOOTHING = 0.99;
    let TIMER_CIRCLE_RADIUS = 15;
    let TIMER_CIRCLE_WIDTH = 2;
    let TIMER_FONT_SIZE = 37;
    let TIMER_COLOR = 'color';
    import * as colors from '$lib/colors.js';
    let PERIOD = 1.0;
    let PHASE = 0.0;
    function get_timer_frac(node, time) {
        return (time - node.phase + PERIOD) % PERIOD / PERIOD;
    }
    function draw() {
        let time = performance.now() / 1000.0;
        if (!ctx) return;
        let color;
        let color_string;
        let inverse_color;
        let inverse_color_string;

        // recalculate the time_elapsed
        if (!pause_timer) {
            time_elapsed = Math.floor((performance.now() - wpm_timer_start) / 1000);
        }
        // Calculate FPS with smoothing
        const currentTime = performance.now();
        const deltaTime = currentTime - lastFrameTime;
        const currentFps = 1000 / deltaTime;
        fps = fps * FPS_SMOOTHING + currentFps * (1 - FPS_SMOOTHING);
        fps = Math.min(100, fps);
        lastFrameTime = currentTime;

        ctx.clearRect(0, 0, canvas_element.width, canvas_element.height);
        
        let now = performance.now() / 1000.0;
        //
        let node = {
            letter: 'a',
            phase: PHASE,
        };
        

        //
        const timerFrac = get_timer_frac(node, time);
        color = colors.color_from_letter(node.letter);
        color_string = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 1.0)`;
        inverse_color = [255 - color[0], 255 - color[1], 255 - color[2]];
        inverse_color_string = `rgba(${inverse_color[0]}, ${inverse_color[1]}, ${inverse_color[2]}, 1.0)`;
        let timer_font_size = TIMER_FONT_SIZE;
        let timer_radius = TIMER_CIRCLE_RADIUS;
        if (node.letter === 'm' || node.letter === 'w') {
        timer_radius *= 1.15;
        }
        const centerX = 100;
        const centerY = 100;
        
        // Draw letter or box
        ctx.beginPath();
        // ctx.fillStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${1 - Math.pow(timerFrac, 1.0)})`;
        ctx.fillStyle = color_string;
        if (node.letter === '$') {
        // Draw stop square
        const stop_square_size = 17;
        ctx.fillRect(centerX - stop_square_size/2, centerY - stop_square_size/2, stop_square_size, stop_square_size);
        } else {
        // Draw letter
        // let letter = node.letter !== ' ' ? node.letter : '_';
        ctx.font = `${timer_font_size}px verdana, helvetica, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(node.letter, centerX, centerY);
        }

        // ctx.globalCompositeOperation = 'xor';
        // Draw timer arc
        ctx.beginPath();
        ctx.arc(centerX, centerY, timer_radius, 0, 2 * Math.PI * timerFrac);
        // ctx.strokeStyle = color_string;
        // ctx.strokeStyle = inverse_color_string;
        // ctx.strokeStyle = 'white';
        // ctx.strokeStyle = `rgba(255, 255, 255, ${timerFrac*0.9+0.1})`;
        if (TIMER_COLOR === 'color') {
        ctx.strokeStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${timerFrac*0.9+0.1})`;
        } else if (TIMER_COLOR === 'white') {
        ctx.strokeStyle = `rgba(255, 255, 255, ${timerFrac*0.9+0.1})`;
        } else {
        throw new Error("Invalid TIMER_COLOR: " + TIMER_COLOR);
        }
        ctx.lineWidth = TIMER_CIRCLE_WIDTH;
        ctx.stroke();
        ctx.closePath();
        //
        requestAnimationFrame(draw);
    }
    function playClickSound() {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();

        // Create an oscillator
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();

        // Set the oscillator frequency for a "click" sound
        oscillator.frequency.value = 1000; // Frequency in Hz (adjust as needed)

        // Set the gain envelope to make it very short (click-like)
        gainNode.gain.setValueAtTime(1, audioContext.currentTime); // Start at full volume
        gainNode.gain.exponentialRampToValueAtTime(0.001, audioContext.currentTime + 0.05); // Fade quickly

        // Connect oscillator to the gain, and then to the audio context
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);

        // Start and stop the oscillator quickly
        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 0.05); // 50ms for a short click
    }

    function click(event) {
        console.log("click");
        if (play_click_sound) {
            playClickSound();
        }
        let time;
        if (event && event.timeStamp) {
            time = event.timeStamp / 1000.0;
        } else {
            time = performance.now() / 1000.0;
        }

        //
        let delay = time - PHASE;
        // map to [-period/2, period/2]
        delay = ((delay + PERIOD*1.5) % PERIOD) - PERIOD/2.0;
        console.log({time, delay});
    }
    onMount(async () => {
        ctx = canvas_element.getContext('2d');
        
        // Get the device pixel ratio
        const dpr = window.devicePixelRatio || 1;
        // Set the canvas size accounting for device pixel ratio
        const rect = canvas_element.getBoundingClientRect();
        canvas_element.width = rect.width * dpr;
        canvas_element.height = rect.height * dpr;
        // Scale the canvas context
        ctx.scale(dpr, dpr);
        // Set the CSS size
        canvas_element.style.width = `${rect.width}px`;
        canvas_element.style.height = `${rect.height}px`;
        // start animation loop
        canvas_element.tabIndex = 0; // Make canvas focusable
        document.addEventListener('keydown', (event) => {
            if (event.code === 'Space') {
                const activeTag = document.activeElement.tagName;
                const isTextInput = (activeTag === 'INPUT' || activeTag === 'TEXTAREA') && document.activeElement.type !== 'checkbox';
                if (!isTextInput) {
                    click(event);
                    event.preventDefault();
                }
            }
        });
        //
        draw();
    });

</script>

<div class="flex flex-col h-full relative box-border">
    <canvas bind:this={canvas_element} class="h-full w-full bg-black"></canvas>
    <div class="absolute top-4 right-6 flex gap-8 text-white text-2xl">
        <label class="flex items-center gap-3">
            <input type="checkbox" bind:checked={blink_to_click} class="w-6 h-6"/>
            Blink to click
        </label>
        <label class="flex items-center gap-3">
            <input type="checkbox" bind:checked={developer_visualizer} class="w-6 h-6"/>
            Debug
        </label>
        <label class="flex items-center gap-3">
            <input type="checkbox" bind:checked={play_click_sound} class="w-6 h-6"/>
            Click sound
        </label>
        <label class="flex items-center gap-3">
            <input type="checkbox" bind:checked={show_boxes} class="w-6 h-6"/>
            Boxes
        </label>
        <label class="flex items-center gap-3">
            <input type="checkbox" bind:checked={use_visual_tutor} class="w-6 h-6"/>
            Tutor
        </label>
    </div>
</div>

{#if blink_to_click}
    <Eye on:blink={click}/>
{/if}
