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
let FIRST_BOX_HEIGHT = 1350;
let BOX_WIDTH = 60;
let TIMER_CIRCLE_RADIUS = 14;
let TIMER_PERIOD = 1.000;
// likelihood parameters
let mu_delay = 0.055;
let sigma = 0.028;
let outliers = 3.0 / 100.0;
// my spacing: mu=0.055, sigma=0.028, outliers=3/100
// my blinking: mu=0.150, sigma=0.040, outliers=3/100 (confirmed xx) (and twice again closely confirmed; do not lightly change)
// (delay down to 115ms)

// mu_delay = 0.150;
// sigma = 0.040;
// outliers = 3.0 / 100.0;

const visibility_threshold = Math.log(0.015);
// let TIMER_PERIOD = 0.7;
// let BAR_PERIOD = 1.0; // 2 seconds
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
true_root.letter = true_root.val[true_root.val.length - 1];
true_root.period = TIMER_PERIOD;
true_root.offset = 0;
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
$: best_delay_stats = best_node.timer_fracs
$: best_delay_mean = best_delay_stats.reduce((a, b) => a + b, 0) / best_delay_stats.length
$: best_delay_std = Math.sqrt(best_delay_stats.reduce((a, b) => a + (b - best_delay_mean)**2, 0) / best_delay_stats.length)
$: best_delay_outliers = best_delay_stats.filter(x => Math.abs(x - best_delay_mean) > 3 * best_delay_std).length
$: best_delay_stats_no_outliers = best_delay_stats.filter(x => Math.abs(x - best_delay_mean) <= 3 * best_delay_std)
$: best_delay_mean_no_outliers = best_delay_stats_no_outliers.reduce((a, b) => a + b, 0) / best_delay_stats_no_outliers.length
$: best_delay_std_no_outliers = Math.sqrt(best_delay_stats_no_outliers.reduce((a, b) => a + (b - best_delay_mean_no_outliers)**2, 0) / best_delay_stats_no_outliers.length)

function handleBlink(event) {
    console.log("Blink event:", event);
    click(event);
}

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
        // const box_width_multiplier = 0.5 + Math.log(numChildren);
        const box_width_multiplier = 0.5;
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


let get_timer_frac = (node, time) => {

    return (time - node.offset + node.period) % node.period / node.period;
}

function draw() {
    let time = performance.now() / 1000.0;
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
        // Draw rectangle with gradient
        const gradient = ctx.createLinearGradient(
            get(node.tweened_location).x - get(left_offset),
            get(node.tweened_location).y,
            get(node.tweened_location).x - get(left_offset),
            get(node.tweened_location).y + get(node.tweened_size_height)
        );
        gradient.addColorStop(0, 'rgba(173, 216, 230, 0.6)'); // Light at top
        gradient.addColorStop(0.5, 'rgba(73, 116, 230, 0.6)'); // Darker in middle
        gradient.addColorStop(1, 'rgba(173, 216, 230, 0.6)'); // Light at bottom

        ctx.beginPath();
        ctx.rect(
            get(node.tweened_location).x - get(left_offset),
            get(node.tweened_location).y,
            get(node.tweened_size_width),
            get(node.tweened_size_height)
        );
        ctx.fillStyle = gradient;
        // ctx.fill();
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 1;
        // ctx.stroke();
        ctx.closePath();
    });
    
    // Draw connections 
    visibleNodes.forEach(node => {
        if (node.parent) {
            const startX = get(node.tweened_location).x + get(node.tweened_size_width)/2 - get(left_offset);
            const startY = get(node.tweened_location).y + get(node.tweened_size_height)/2;
            const endX = get(node.parent.tweened_location).x + get(node.parent.tweened_size_width)/2 - get(left_offset);
            const endY = get(node.parent.tweened_location).y + get(node.parent.tweened_size_height)/2;
            
            // Calculate control points for the curve
            const midY = (startY + endY) / 2;
            const controlPoint1X = startX;
            const controlPoint1Y = midY;
            const controlPoint2X = endX; 
            const controlPoint2Y = midY;

            ctx.beginPath();
            ctx.moveTo(startX, startY);
            ctx.bezierCurveTo(
                controlPoint1X, controlPoint1Y,
                controlPoint2X, controlPoint2Y,
                endX, endY
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
        ctx.fillStyle = 'white';
        ctx.fill();
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.closePath();

        // Draw timer arc
        const timerFrac = get_timer_frac(node, time);
        const threshold = 0.04;
        const is_close = timerFrac < threshold || timerFrac > 1 - threshold;
        ctx.beginPath();
        ctx.arc(centerX, centerY, TIMER_CIRCLE_RADIUS, 0, 2 * Math.PI * timerFrac);
        // Transition from red to green using HSL, with bright flash at completion
        const hue = 0;
        const lightness = 40 + is_close * 20; // Start at 50% and go to 100%
        const saturation = 80 + is_close * 20; // Start at 50% and go to 100%
        const color = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
        ctx.strokeStyle = color;
        ctx.lineWidth = 4;
        ctx.stroke();
        ctx.closePath();

        // Draw letter
        ctx.beginPath();
        ctx.fillStyle = 'black';
        ctx.font = '28px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(node.letter, centerX, centerY);
        ctx.closePath();
    });


    animationFrameId = requestAnimationFrame(draw);
}

function get_widths(ps, C, sigma) {
    return ps.map(p => Math.sqrt(2 * sigma * sigma * Math.max(0, Math.log(p) - C)));
}
function find_C(ps, target_width, sigma) {
    console.log("ps:", ps);
    let C_min = Math.log(0.01) - 1000;
    let C_max = Math.log(1.00) + 1;
    let widths;
    let sum_widths;
    while (C_max - C_min > 1e-5) {
        let C = (C_min + C_max) / 2;
        widths = get_widths(ps, C, sigma);
        sum_widths = widths.reduce((a, b) => a + b, 0);
        if (sum_widths > target_width) {
            C_min = C;
        } else {
            C_max = C;
        }
    }
    widths = widths.map(w => w / sum_widths * target_width);
    return widths;
}

function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}

let socket;
let bin_probs_chart_data = [];
let offsets_chart_data = [];
function click(event) {
    let time = event.timeStamp / 1000.0;
    if (!socket) {
        console.log("No socket; can't click");
        return;
    }
    // push likelihoods
    Object.values(true_root.registry).forEach(node => {
        if (node.is_visible) {
            let time_remainder = time % node.period;
            let delay = time_remainder - node.offset;
            // map to [-period/2, period/2]
            delay = ((delay + node.period*1.5) % node.period) - node.period/2.0;
            let dx = delay - mu_delay;
            let gaussian_log_likelihood = - Math.log(Math.sqrt(2 * Math.PI)*sigma) - dx * dx / (2 * sigma * sigma)
            // we had 2 outliers in 100 samples, so
            const uniform_likelihood = Math.log( outliers );
            let timer_likelihood = Math.max(gaussian_log_likelihood, uniform_likelihood)
            problogic.push_likelihood(node, timer_likelihood, delay);
        }
    });

    console.log("updating trie after updating likelihoods");
    problogic.run_func_w_timing(problogic.update_trie, [true_root, false, 0, lm_request_queue, pDATA, LM, visibility_threshold, tokenizer]);
    problogic.run_func_w_timing(problogic.calc_posteriors, [true_root]); // determine the normalizing constant
    pDATA = true_root.post_Z;
    
    socket.send(JSON.stringify({type: 'set_queue', content: lm_request_queue.queue}));
    
    let offsets = [];
    // first pass to determine visibility
    for (let node of Object.values(true_root.registry)) {
        let normalized_posterior = node.post_Z - pDATA;
        node.is_visible = normalized_posterior > visibility_threshold;
    }
    // second pass to get selection probabilities
    for (let node of Object.values(true_root.registry)) {
        if (node.is_visible) {
            let selection_prob = Math.exp(node.post_Z - pDATA);
            for (let child of node.children) {
                if (child.is_visible) {
                    selection_prob -= Math.exp(child.post_Z - pDATA);
                }
            }
            offsets.push([node, node.val, selection_prob]);
        }
    }
    // set the offsets for visible nodes
    let widths = find_C(offsets.map(o => Math.max(0, o[2])), TIMER_PERIOD/2.0, sigma);
    // create a random permutation of widths.length elements
    let random_order = shuffleArray(Array.from({length: widths.length}, (_, i) => i));
    let cum_width = 0;
    for (let i = 0; i < random_order.length; i++) {
        // offsets[random_order[i]][0].offset = Math.random() * TIMER_PERIOD;
        offsets[random_order[i]][0].offset = cum_width + widths[random_order[i]];
        cum_width += 2 * widths[random_order[i]];
    }
    if (Math.abs(cum_width - TIMER_PERIOD) > 1e-5) {
        throw new Error("cum_width != TIMER_PERIOD: " + cum_width + " " + TIMER_PERIOD);
    }

    // create histogram data, of selection probabilities
    let bin_resolution = 1000;
    let bin_width = TIMER_PERIOD / bin_resolution;
    let bins = Array.from({length: bin_resolution + 1}, (_, i) => i * bin_width);
    let bin_probs = Array.from({length: bins.length}, () => 0);
    for (let i = 0; i < bins.length; i++) {
        let bin = bins[i];
        bin_probs[i] = 0;
        for (let offset_tuple of offsets) {
            let node = offset_tuple[0];
            let delay = bin - node.offset;
            delay = ((delay + node.period*1.5) % node.period) - node.period/2.0;
            let dx = delay - mu_delay;
            let gaussian_log_likelihood = - Math.log(Math.sqrt(2 * Math.PI)*sigma) - dx * dx / (2 * sigma * sigma)
            bin_probs[i] += offset_tuple[2] * Math.exp(gaussian_log_likelihood);
        }
    }
    console.log("Bin probs:", bin_probs);
    bin_probs_chart_data = bin_probs.map((prob, i) => ({x: bins[i] / TIMER_PERIOD, y: prob / Math.max(...bin_probs)}));
    offsets_chart_data = offsets.filter(offset => offset[2] > Math.exp(visibility_threshold)).map((offset, i) => ({x: ((offset[0].offset + mu_delay) % TIMER_PERIOD) / TIMER_PERIOD, text: offset[1]}));
    setLocations(true_root);
    // console.log("Offsets:", offsets);
}

let lm_request_queue = {
    // data structure: sorted list of (key,priority) pairs; associated set for membership checks
    queue: {},  // key,priority pairs
    least_priority_key: null,
    least_priority_value: Infinity,
    has(key) {
        return this.queue.hasOwnProperty(key);
    },
    remove(key) {
        delete this.queue[key];
        if (key === this.least_priority_key) {
            this.refresh_least_priority();
        }
    },
    remove_lowest_priority() {
        this.remove(this.least_priority_key);
    },
    insert(key, priority) {
        this.queue[key] = priority;
        if (priority < this.least_priority_value) {
            this.least_priority_key = key;
            this.least_priority_value = priority;
        }
    },
    update(key, priority) {
        let old_priority = this.queue[key];
        this.queue[key] = priority;
        // are we the new lowest priority?
        if (priority < this.least_priority_value) {
            this.least_priority_key = key;
            this.least_priority_value = priority;
        // were we the old lowest priority? if so, rescan to find the new lowest priority
        } else if (key === this.least_priority_key) {
            this.refresh_least_priority();
        }
    },
    refresh_least_priority() {
        let entries = Object.entries(this.queue);
        if (entries.length === 0) {
            this.least_priority_key = null;
            this.least_priority_value = Infinity;
            return;
        }
        let min_entry = entries.reduce((min, curr) => curr[1] < min[1] ? curr : min);
        this.least_priority_key = min_entry[0];
        this.least_priority_value = min_entry[1];
    },
    get_length() {
        return Object.keys(this.queue).length;
    }
};
let lm_response_queue = [];
onMount(async () => {
    // load the tokenizer
    tokenizer = await AutoTokenizer.from_pretrained("llama/llama", {local_files_only: true});
    console.log("Tokenizer loaded");
    true_root.tokenization = problogic.get_tokenization(true_root.val, tokenizer);
    console.log("Tokenization for root:", true_root.tokenization);

    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');
    
    // Set canvas size
    canvas.width = 6000;
    canvas.height = FIRST_BOX_HEIGHT;
    
    // Start animation loop
    draw();

    window.addEventListener('keydown', (event) => {
        if (event.code === 'Space') {
            click(event);
            event.preventDefault();
        }
    });
    window.addEventListener('click', (event) => {
        click(event);
        event.preventDefault(); 
    });

    const urlParams = new URLSearchParams(window.location.search);
    const is_local = urlParams.get('serve_local') === 'true';
    console.log("is_local:", is_local);

    // socket = new WebSocket('ws://localhost:8000/ws');
    // socket = new WebSocket('ws://localhost:8001/ws');
    // socket = new WebSocket('ws://8.34.124.122:20425/ws');
    if (is_local) {
        socket = new WebSocket('ws://localhost:8000/ws');
    } else {
        socket = new WebSocket('wss://dasher.domainnamefortesting.com:20002/ws');
    }
    
    socket.addEventListener('open', (event) => {
        console.log('WebSocket connection established');
        socket.send(JSON.stringify({type: 'set_queue', content: {[true_root.val]: 0}}));
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
            let lm_response_node = true_root.registry[response.ftp];
            lm_response_node.in_character_model = true;
            // remove from lm_request_queue
            if (lm_request_queue.has(response.ftp)) {
                lm_request_queue.remove(response.ftp);
            }
            lm_response_queue.push(lm_response_node);

            problogic.run_func_w_timing(problogic.update_trie, [lm_response_node, true, 0, lm_request_queue, pDATA, LM, visibility_threshold, tokenizer]);
            
            console.log("--------------------------------");
            console.log("FTP:", response.ftp);
            console.log("ROOT:", true_root);
            console.log("Queue:", lm_request_queue.queue);
            console.log("--------------------------------");
            // !Incorrect: this overwrites requests that may be outside of the update branch
            socket.send(JSON.stringify({type: 'set_queue', content: lm_request_queue.queue}));
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

<div class="flex flex-row gap-4">
    <div class="flex items-center gap-2">
        <label for="delay">Delay:</label>
        <input id="delay" type="number" bind:value={mu_delay} step="0.001" class="border rounded px-2 py-1" />
    </div>
    <div class="flex items-center gap-2">
        <label for="deviation">Deviation:</label>
        <input id="deviation" type="number" bind:value={sigma} step="0.001" class="border rounded px-2 py-1" />
    </div>
    <div class="flex items-center gap-2">
        <label for="outliers">Outliers:</label>
        <input id="outliers" type="number" bind:value={outliers} step="0.001" class="border rounded px-2 py-1" />
    </div>
</div>

<Eye on:blink={handleBlink} />

<div class="flex flex-col">
    <div class="text-2xl font-bold">
        {best_string}
    </div>
    <div>
        Delay: {Math.round(best_delay_mean_no_outliers * 1000)}ms ± {Math.round(best_delay_std_no_outliers * 1000)}ms
        Outliers: {best_delay_outliers} / {best_delay_stats.length}
    </div>
</div>

<canvas id="canvas" class="w-[6000px] h-[{FIRST_BOX_HEIGHT}px]"></canvas>

<div class="text-2xl font-bold">
    {Math.round(fps)} FPS
</div>

<div class="w-[1000px] h-[500px] relative border-2 border-black padding-4">
    {#if bin_probs_chart_data && bin_probs_chart_data.length > 0}
        {#each bin_probs_chart_data as point}
            <div 
                class="absolute w-1 h-1 bg-blue-500" 
                style="left: {point.x * 1000}px; bottom: {point.y * 500}px;"
            />
        {/each}
    {/if}
    {#if offsets_chart_data && offsets_chart_data.length > 0}
        {#each offsets_chart_data as point}
            <div 
                class="absolute transform rotate-90 whitespace-nowrap origin-left"
                style="left: {point.x * 1000}px; top: 500px;"
            >
                {point.text}
            </div>
            <div
                class="absolute w-[2px] h-[500px] bg-red-500"
                style="left: {point.x * 1000}px; top: 0px;"
            />
        {/each}
    {/if}
</div>