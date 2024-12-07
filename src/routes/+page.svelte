<script>
import { onMount } from 'svelte';
import { tweened } from 'svelte/motion';
import { linear, cubicOut, quadInOut, sineInOut } from 'svelte/easing';
import { get } from 'svelte/store';
import Eye from './eye.svelte';
import * as problogic from '$lib/tokenizer.js';
import { env, AutoTokenizer } from "@huggingface/transformers";
env.allowLocalModels = true;
let LM = {};
let FIRST_BOX_HEIGHT = 800;
let BOX_WIDTH = 60;
let TIMER_CIRCLE_RADIUS = 14;
let TIMER_PERIOD = 1.4;
// let TIMER_PERIOD = 0.7;
let BAR_PERIOD = 1.0; // 2 seconds
let fps = 60;
let lastFrameTime = performance.now();
const FPS_SMOOTHING = 0.99;

let canvas;
let ctx;
let animationFrameId;

let tokenizer;

// $: bar_height = (time/BAR_PERIOD % 1) * FIRST_BOX_HEIGHT;

let true_root = {prior: 0, likelihood: 0, children: [], prior_ill: 0, ever_visible_parent: false};
true_root.val = ".";
true_root.val = `Text entry research typically pits one entry method against another. Thus,
entry method is the controlled variable, and it is manipulated over two or more levels,
for example, Multitap vs. Letterwise in an experiment comparing text entry techniques for
mobile phones [2], or Qwerty vs. Opti in an experiment comparing soft keyboard layouts [3].

Allowing`;
true_root.val = ".";
true_root.force_space = true;
true_root.tokenization = problogic.get_tokenization(true_root.val);
console.log("Tokenization for root:", true_root.tokenization);
true_root.letter = true_root.val[true_root.val.length - 1];
true_root.period = TIMER_PERIOD;
true_root.offset = Math.random() * true_root.period;
true_root.height = 0;
true_root.y_relative_bottom = 0;
true_root.y_relative_top = 1;
true_root.trie = true_root;
true_root.registry = {};
true_root.registry[true_root.val] = true_root;
true_root.timer_fracs = [];
let left_offset = tweened(0, { duration: 4000, easing: linear });

function get_best_node(node) {
    if (pDATA !== undefined && node.tweened_location && get(node.tweened_size_height) > 0.99 * FIRST_BOX_HEIGHT) {
        let proposed_offset = get(node.tweened_location).x - 400;
        if (proposed_offset > get(left_offset)) {
            left_offset.set(proposed_offset);
        }
    }
    if (node.children.length === 0) {
        return node
    }
    let best_child = node.children[0];
    for (const child of node.children) {
        if (child.post_Z > best_child.post_Z) {
            best_child = child;
        }
    }
    return get_best_node(best_child);
}
$: best_node = get_best_node(true_root)
$: best_string = best_node.val

function handleBlink(event) {
    console.log("Blink event:", event);
    click();
}

let visibility_threshold = Math.log(0.03);
let pDATA = 0;
const TWEEN_DURATION = 300;
const TWEEN_EASING = linear;

function setLocations(node, loc = null, size_height = FIRST_BOX_HEIGHT, size_width = BOX_WIDTH) {
    if (loc === null) {
        loc = { x: 0, y: 0 };
    }
    node.location = { ...loc };
    if (!node.tweened_location) {
        node.tweened_location = tweened(loc, { duration: TWEEN_DURATION, easing: TWEEN_EASING });
        node.tweened_size_height = tweened(size_height, { duration: TWEEN_DURATION, easing: TWEEN_EASING });
        node.tweened_size_width = tweened(size_width, { duration: TWEEN_DURATION, easing: TWEEN_EASING });
    } else {
        node.tweened_location.set(loc);
        node.tweened_size_height.set(size_height);
        node.tweened_size_width.set(size_width);
    }
    node.size_height = size_height;
    node.size_width = size_width;
    node.is_visible = true;
    let visible_children = node.children.filter(child => child.is_visible);

    if (visible_children.length > 0) {
        node.ever_visible_parent = true;
        const numChildren = visible_children.length;
        const box_width_multiplier = 0.5 + Math.log(numChildren);
        visible_children.forEach((child, index) => {
            setLocations(child, {
                x: node.location.x + node.size_width,
                y: node.location.y + node.size_height * child.y_relative_bottom,
            }, node.size_height * (child.y_relative_top - child.y_relative_bottom), box_width_multiplier * BOX_WIDTH);
        });
    }

    if (node.height === 0) {
        true_root = true_root;
    }
}
setLocations(true_root);


function get_time() {
    return Date.now() / 1000.0;
}
let get_timer_frac = (node, time) => {
    return (time - node.offset + node.period) % node.period / node.period;
}

function draw() {
    let time = get_time();
    if (!ctx) return;
    
    // Calculate FPS with smoothing
    const currentTime = performance.now();
    const deltaTime = currentTime - lastFrameTime;
    const currentFps = 1000 / deltaTime;
    fps = fps * FPS_SMOOTHING + currentFps * (1 - FPS_SMOOTHING);
    fps = Math.min(100, fps);
    lastFrameTime = currentTime;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const visibleNodes = Object.values(true_root.registry).filter(node => node.is_visible);
    

    // Draw nodes
    visibleNodes.forEach(node => {
        // Draw rectangle
        ctx.beginPath();
        ctx.rect(
            get(node.tweened_location).x - get(left_offset),
            get(node.tweened_location).y,
            get(node.tweened_size_width),
            get(node.tweened_size_height)
        );
        ctx.fillStyle = 'lightblue';
        ctx.fill();
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.closePath();
    });
    
    // Draw connections 
    visibleNodes.forEach(node => {
        if (node.parent) {
            ctx.beginPath();
            ctx.moveTo(
                get(node.tweened_location).x + get(node.tweened_size_width)/2 - get(left_offset),
                get(node.tweened_location).y + get(node.tweened_size_height)/2
            );
            ctx.lineTo(
                get(node.parent.tweened_location).x + get(node.parent.tweened_size_width)/2 - get(left_offset),
                get(node.parent.tweened_location).y + get(node.parent.tweened_size_height)/2
            );
            ctx.strokeStyle = 'black';
            ctx.lineWidth = 2;
            ctx.stroke();
            ctx.closePath();
        }
    });

    // Draw timer circles
    visibleNodes.forEach(node => {
        const centerX = get(node.tweened_location).x + get(node.tweened_size_width)/2 - get(left_offset);
        const centerY = get(node.tweened_location).y + get(node.tweened_size_height)/2;
        
        // Draw circle background
        ctx.beginPath();
        ctx.arc(centerX, centerY, TIMER_CIRCLE_RADIUS, 0, 2 * Math.PI);
        ctx.fillStyle = 'lightblue';
        ctx.fill();
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.closePath();

        // Draw timer arc
        ctx.beginPath();
        ctx.arc(centerX, centerY, TIMER_CIRCLE_RADIUS, 0, 2 * Math.PI * get_timer_frac(node, time));
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 4;
        ctx.stroke();
        ctx.closePath();

        // Draw letter
        ctx.beginPath();
        ctx.fillStyle = 'black';
        ctx.font = '20px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(node.letter, centerX, centerY);
        ctx.closePath();
    });


    animationFrameId = requestAnimationFrame(draw);
}

let socket;
function click() {
    let time = get_time();
    if (!socket) {
        console.log("No socket; can't click");
        return;
    }
    // push likelihoods
    Object.values(true_root.registry).forEach(node => {
        if (node.is_visible) {
            let timerFrac = get_timer_frac(node, time);
            timerFrac = (timerFrac + 0.5) % 1.0;
            // decent heuristic
            // let timer_likelihood = - Math.pow(0.5 - timerFrac, 2) * 40.0;
            // BLINKING
            // t = 1.400s
            // empirically derived from a sample of 100 blinks
            // mu = 0.565; sigma = 0.037;
            // a different sample of 153 gave mu=0.572; sigma=0.044
            // the gaussian has log_likelihood of (-dx^2/(2sigma^2)) up to a factor constant wrt dx
            // const mu = 0.572;
            // const sigma = 0.044;
            // outliers = 3.0 / 100.0

            // SPACING
            // t = 0.700s
            // mu = 0.619; sigma = 0.075;
            // outliers = 6.0 / 152.0
            const mu = 0.570;
            const sigma = 0.044;
            const outliers = 3.0 / 100.0;
            let dx = timerFrac - mu;
            let gaussian_log_likelihood = - Math.log(Math.sqrt(2 * Math.PI)*sigma) - dx * dx / (2 * sigma * sigma)
            // we had 2 outliers in 100 samples, so
            const uniform_likelihood = Math.log( outliers );
            let timer_likelihood = Math.max(gaussian_log_likelihood, uniform_likelihood)
            problogic.push_likelihood(node, timer_likelihood, timerFrac);
            node.offset = Math.random() * node.period;
        }
    });

    let queue = [];
    console.log("updating trie after updating likelihoods");
    problogic.run_func_w_timing(problogic.update_trie, [true_root, false, 0, queue, pDATA, LM, visibility_threshold, tokenizer]);
    problogic.run_func_w_timing(problogic.calc_posteriors, [true_root]);
    pDATA = true_root.post_Z;
    
    queue.sort((a, b) => b[1] - a[1]);
    console.log("Queue:", queue);
    socket.send(JSON.stringify({type: 'set_queue', content: queue.map(q => q[0])}));
    
    for (let node of Object.values(true_root.registry)) {
        node.is_visible = node.post_Z - pDATA > visibility_threshold;
    }
    setLocations(true_root);
}


onMount(async () => {
    // load the tokenizer
    tokenizer = await AutoTokenizer.from_pretrained("llama/llama", {local_files_only: true});

    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');
    
    // Set canvas size
    canvas.width = 6000;
    canvas.height = 6000;
    
    // Start animation loop
    draw();

    window.addEventListener('keydown', (event) => {
        if (event.code === 'Space') {
            click();
            event.preventDefault();
        }
    });

    // socket = new WebSocket('ws://localhost:8000/ws');
    // socket = new WebSocket('ws://localhost:8001/ws');
    socket = new WebSocket('ws://38.224.253.4:20801/ws');
    
    socket.addEventListener('open', (event) => {
        console.log('WebSocket connection established');
        socket.send(JSON.stringify({type: 'set_queue', content: [true_root.val]}));
        setTimeout(() => {
            socket.send(JSON.stringify({type: 'echo', content: 'Hello, World!'}));
        }, 200);
    });

    socket.addEventListener('message', async (event) => {
        let start = performance.now();
        let response = JSON.parse(event.data);
        let end = performance.now();
        if (response.type === 'processed') {
            console.log("JSON parsing took", end - start, "ms");
            await new Promise(resolve => setTimeout(resolve, 5));
            LM[response.ftp] = {"probs": response.probs, "cum": response.cum};
            true_root.registry[response.ftp].in_character_model = true;
            let queue = [];

            problogic.run_func_w_timing(problogic.update_trie, [true_root.registry[response.ftp], true, 0, queue, pDATA, LM, visibility_threshold, tokenizer]);
            // TODO
            // problogic.run_func_w_timing(problogic.calc_posteriors, [true_root]);
            pDATA = true_root.post_Z;
            setLocations(true_root);
            
            queue.sort((a, b) => b[1] - a[1]);
            console.log("--------------------------------");
            console.log("FTP:", response.ftp);
            console.log("ROOT:", true_root);
            console.log("Queue:", queue);
            console.log("--------------------------------");
            socket.send(JSON.stringify({type: 'set_queue', content: queue.map(q => q[0])}));
        }
    });

    socket.addEventListener('close', (event) => {
        console.log('WebSocket connection closed');
    });

    socket.addEventListener('error', (event) => {
        console.error('WebSocket error:', event);
    });

    return () => {
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
        }
    };
});
</script>

<Eye on:blink={handleBlink} />

<div class="flex flex-col">
    <div class="text-2xl font-bold">
        {Math.round(fps)} FPS
    </div>
    <div>
        {JSON.stringify(best_node.timer_fracs)}
    </div>
    <div class="text-2xl font-bold">
        {best_string}
    </div>
</div>

<canvas id="canvas" class="w-[6000px] h-[6000px]"></canvas>
