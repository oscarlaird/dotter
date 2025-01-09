<script>

// experiments
// 1.
// large font ~90 large boxes ~45 high contrast w/ black and white
// use 10->100% opacity of timer to reduce clutter
// 2. 
// use medium-small font (looks more elegant) / medium-small boxes
// use bold font (looks less elegant)
// use same color for timer (looks more elegant)
// remove circle background (connections meet so looks more elegant)
// use a thin timer line to make it less cluttered (looks more elegant)
// notes:
// in both cases it seems better to use a thin timer line, but make it larger
// the first way (though less elegant) certainly feels faster
// and it would allow us to kern
// doing the timer line with xor compositeMode does not help (better just use black background; white timer line)
// 3.
// tried to dispense with timers entirely and react to flashes or a color_value transition; this was hard to time
// only got 220ms delay; 80ms stddev; worse than blinking
// 4. Can we make something come down from top and bottom and meet in the middle of the letter?

import { onMount } from 'svelte';
import { tweened } from 'svelte/motion';
import { tick } from 'svelte';
import { linear, cubicOut, quadInOut, sineInOut } from 'svelte/easing';
import { get } from 'svelte/store';
import Eye from './eye.svelte';
import * as problogic from '$lib/tokenizer.js';
import * as colors from '$lib/colors.js';
import { env, AutoTokenizer } from "@huggingface/transformers";
import phrases_text from '$lib/phrases.txt?raw';
let phrases = phrases_text.split('\n').map(line => line.trim()).filter(line => line.length > 0);
let session_active = true;
function random_phrase() {
    return phrases[Math.floor(Math.random() * phrases.length)];
}
let target_phrase = random_phrase();

env.allowLocalModels = true;
let LM = {};
let FIRST_BOX_HEIGHT;

let canvas_element;
let BOX_WIDTH = 37;
let BOX_WIDTH_CHILDREN_MULTIPLIER = 1.0;
// BOX_WIDTH = 100;
let TIMER_CIRCLE_RADIUS = 15;
// let TIMER_CIRCLE_RADIUS = 13;

let TIMER_CIRCLE_WIDTH = 2;
// let TIMER_CIRCLE_WIDTH = 4;

let TIMER_FONT_SIZE = 37;
// let TIMER_FONT_SIZE = 70;

let TIMER_COLOR = 'color';
// let TIMER_COLOR = 'white';
let TIMER_PERIOD = 1.000;

// likelihood parameters
let mu_delay_ms = 35;
let sigma_ms = 30;
let outliers_perc = 3.0;
$: mu_delay = mu_delay_ms / 1000.0;
$: sigma = sigma_ms / 1000.0;
$: outliers = outliers_perc / 100.0;

let toggle_settings = false;
let wpm_timer_start;
let time_elapsed;
let pause_timer = true;
let awaiting_first_gesture = true;
$: n_words = best_string.split(' ').length - 1;
$: wpm = (60 * n_words / time_elapsed);
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

let ctx;
let animationFrameId;

let tokenizer;

$: if (toggle_settings) {
    setTimeout(async () => {
        await tick();
        FIRST_BOX_HEIGHT = canvas_element.clientHeight;
        canvas_element.width = canvas_element.clientWidth;
        canvas_element.height = FIRST_BOX_HEIGHT;
    }, 0);
}
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
// true_root.period = TIMER_PERIOD;
true_root.offset = 0;
true_root.height = 0;
true_root.y_relative_bottom = 0;
true_root.y_relative_top = 1;
true_root.registry = {};
true_root.timer_fracs = [];
let left_offset_tweening_params = { duration: 4000, easing: linear };
let left_offset = tweened(0, left_offset_tweening_params);
let frozen_root = JSON.stringify(true_root);
true_root.trie = true_root;
true_root.registry[true_root.val] = true_root;

function reset() {
    wpm_timer_start = performance.now();
    time_elapsed = 0;
    pause_timer = true;
    awaiting_first_gesture = true;
    confirmed = false;
    let new_root = JSON.parse(frozen_root);
    new_root.registry[new_root.val] = new_root;
    new_root.trie = new_root;
    new_root.tokenization = problogic.get_tokenization(new_root.val, tokenizer);
    true_root = new_root;
    socket.send(JSON.stringify({type: 'set_queue', content: [[true_root.val, 0, true_root.tokenization]]}));
    left_offset = tweened(0, left_offset_tweening_params);
    target_phrase = random_phrase();
    speak(target_phrase);
}

let confirmed = false;
let confirm_threshold = Math.log(0.95);
function speak(text) {
    if (!('speechSynthesis' in window)) {
        console.log('Text-to-speech not supported in this browser');
        return;
    }

    console.log("speaking:", text);
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.6;
    utterance.pitch = 1.0;
    utterance.volume = 0.9;

    const voices = window.speechSynthesis.getVoices();
    const preferredVoice = voices.find(voice => voice.name.includes("Google") || voice.name.includes("Microsoft"));
    
    if (preferredVoice) {
        utterance.voice = preferredVoice;
    }

    window.speechSynthesis.speak(utterance);
}
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
    if (best_child.letter === "$" && !confirmed && (best_child.post_Z - pDATA) > confirm_threshold) {
        confirmed = true;
        pause_timer = true;
        speak(best_child.val.slice(2, -1));
        socket.send(JSON.stringify({
            type: 'log',
            content: {
                best_string: best_string,
                target_phrase: target_phrase,
                timer_fracs: best_delay_stats,
                wpm_timer_start: wpm_timer_start,
                time_elapsed: time_elapsed,
                wpm: wpm
            }
        }))
        if (session_active) {
            setTimeout(() => {
                reset();
            }, 2000);
        }
    }
    return get_best_node(best_child);
}

function longest_common_prefix(s1, s2) {
    let i = 0;
    while (i < s1.length && i < s2.length && s1[i] === s2[i]) {
        i++;
    }
    return s1.slice(0, i);
}
$: best_node = get_best_node(true_root)
$: best_string = best_node.val
$: best_string_agrees_with_target = longest_common_prefix(best_string.slice(2), target_phrase)
$: uncompleted_suffix = target_phrase.slice(best_string_agrees_with_target.length)
$: best_delay_stats = best_node.timer_fracs
$: best_delay_mean = best_delay_stats.reduce((a, b) => a + b, 0) / best_delay_stats.length
$: best_delay_std = Math.sqrt(best_delay_stats.reduce((a, b) => a + (b - best_delay_mean)**2, 0) / best_delay_stats.length)
$: empirical_outliers = best_delay_stats.filter(x => Math.abs(x - best_delay_mean) > 3 * best_delay_std).length
$: empirical_outliers_perc = empirical_outliers / best_delay_stats.length * 100
$: best_delay_stats_no_outliers = best_delay_stats.filter(x => Math.abs(x - best_delay_mean) <= 3 * best_delay_std)
$: empirical_delay = best_delay_stats_no_outliers.reduce((a, b) => a + b, 0) / best_delay_stats_no_outliers.length
$: empirical_std = Math.sqrt(best_delay_stats_no_outliers.reduce((a, b) => a + (b - empirical_delay)**2, 0) / best_delay_stats_no_outliers.length)

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
    node.ever_visible = true;
    let visible_children = node.children.filter(child => child.is_visible);

    if (visible_children.length > 0) {
        node.ever_visible_parent = true;
        const numChildren = visible_children.length;
        const box_width_multiplier = 1 + BOX_WIDTH_CHILDREN_MULTIPLIER*Math.log(numChildren);
        // const box_width_multiplier = 0.5;
        visible_children.forEach((child, index) => {
            setLocations(child, {
                x: node.location.x + node.size_width,
                y: node.location.y + node.size_height * child.y_relative_bottom,
            }, node.size_height * (child.y_relative_top - child.y_relative_bottom), box_width_multiplier * BOX_WIDTH);
        });
        // node.children.forEach((child, index) => {
        //     setLocations(child, {
        //         x: node.location.x + node.size_width,
        //         y: node.location.y + node.size_height * child.y_relative_bottom,
        //     }, node.size_height * (child.y_relative_top - child.y_relative_bottom), box_width_multiplier * BOX_WIDTH);
        // });
    }

    if (node.height === 0) {
        true_root = true_root;
    }
}
setLocations(true_root);


let get_timer_frac = (node, time) => {

    return (time - node.offset + TIMER_PERIOD) % TIMER_PERIOD / TIMER_PERIOD;
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
    
    const visibleNodes = Object.values(true_root.registry).filter(node => node.is_visible);
    

    // Draw nodes
    visibleNodes.forEach(node => {
        color = colors.color_from_letter(node.letter);
        color_string = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 1.0)`;

        ctx.beginPath();
        ctx.rect(
            get(node.tweened_location).x - get(left_offset),
            get(node.tweened_location).y,
            get(node.tweened_size_width),
            get(node.tweened_size_height)
        );
        ctx.strokeStyle = color_string;
        ctx.lineWidth = 1;
        // ctx.stroke();
        ctx.closePath();
    });
    
    // Draw connections 
    visibleNodes.forEach(node => {
        color = colors.color_from_letter(node.letter);
        color_string = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0.5)`;
        const darker_color = `rgba(${Math.floor(color[0] / 2)}, ${Math.floor(color[1] / 2)}, ${Math.floor(color[2] / 2)}, 1.0)`;
        if (node.parent) {
            const startX = get(node.tweened_location).x + get(node.tweened_size_width) - get(left_offset) - TIMER_CIRCLE_RADIUS;
            const startY = get(node.tweened_location).y + get(node.tweened_size_height)/2;
            const endX = get(node.parent.tweened_location).x + get(node.parent.tweened_size_width) - get(left_offset) - TIMER_CIRCLE_RADIUS;
            const endY = get(node.parent.tweened_location).y + get(node.parent.tweened_size_height)/2;
            
            // Calculate control points for the curve
            const midX = (startX + endX) / 2;
            const controlPoint1X = midX;
            const controlPoint1Y = startY;
            const controlPoint2X = midX;
            const controlPoint2Y = endY;

            ctx.beginPath();
            ctx.moveTo(startX, startY);
            ctx.bezierCurveTo(
                controlPoint1X, controlPoint1Y,
                controlPoint2X, controlPoint2Y,
                endX, endY
            );
            // ctx.strokeStyle = color_string;
            ctx.strokeStyle = darker_color;
            ctx.lineWidth = 2;
            ctx.stroke();
            ctx.closePath();
        }
    });

    // Draw timer circles
    visibleNodes.forEach(node => {
        const timerFrac = get_timer_frac(node, time);
        color = colors.color_from_letter(node.letter);
        color_string = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 1.0)`;
        inverse_color = [255 - color[0], 255 - color[1], 255 - color[2]];
        inverse_color_string = `rgba(${inverse_color[0]}, ${inverse_color[1]}, ${inverse_color[2]}, 1.0)`;
        const centerX = get(node.tweened_location).x + get(node.tweened_size_width) - get(left_offset) - TIMER_CIRCLE_RADIUS;
        const centerY = get(node.tweened_location).y + get(node.tweened_size_height)/2;
        
        // Draw letter or box
        ctx.beginPath();
        // ctx.fillStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${1 - Math.pow(timerFrac, 1.0)})`;
        ctx.fillStyle = color_string;
        if (node.letter === '$') {
            // Draw green box
            const boxSize = 20;
            ctx.fillRect(centerX - boxSize/2, centerY - boxSize/2, boxSize, boxSize);
        } else {
            // Draw letter
            // let letter = node.letter !== ' ' ? node.letter : '_';
            ctx.font = `${TIMER_FONT_SIZE}px verdana, helvetica, sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(node.letter, centerX, centerY);
        }
        ctx.closePath();

        // Draw circle background
        // ctx.beginPath();
        // ctx.arc(centerX, centerY, TIMER_CIRCLE_RADIUS, 0, 2 * Math.PI);
        // ctx.strokeStyle = color_string;
        // ctx.lineWidth = 2;
        // ctx.stroke();
        // ctx.closePath();

        // ctx.globalCompositeOperation = 'xor';
        // Draw timer arc
        ctx.beginPath();
        ctx.arc(centerX, centerY, TIMER_CIRCLE_RADIUS, 0, 2 * Math.PI * timerFrac);
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
        // ctx.globalCompositeOperation = 'source-over';


        // // Draw radius line based on timer fraction
        // if (node.timer_fracs.length > 0) {
        //     let last_delay_error = node.timer_fracs[node.timer_fracs.length-1] - mu_delay;
        //     let last_delay_angle = last_delay_error / TIMER_PERIOD * 2 * Math.PI;
        //     ctx.beginPath();
        //     ctx.moveTo(
        //         centerX + 0.7 * TIMER_CIRCLE_RADIUS * Math.cos(last_delay_angle),
        //         centerY + 0.7 * TIMER_CIRCLE_RADIUS * Math.sin(last_delay_angle)
        //     );
        //     ctx.lineTo(
        //         centerX + 1.4 * TIMER_CIRCLE_RADIUS * Math.cos(last_delay_angle),
        //         centerY + 1.4 * TIMER_CIRCLE_RADIUS * Math.sin(last_delay_angle)
        //     );
        //     ctx.strokeStyle = color_string;
        //     ctx.lineWidth = 1;
        //     // ctx.stroke();
        //     // ctx.closePath();
        // }

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
    if (!socket) {
        console.log("No socket; can't click");
        return;
    }
    // push likelihoods
    if (event !== null) {
        if (awaiting_first_gesture) {
            awaiting_first_gesture = false;
            wpm_timer_start = performance.now();
            pause_timer = false;
        }
        let time = event.timeStamp / 1000.0;
        let time_remainder = time % TIMER_PERIOD;
        Object.values(true_root.registry).forEach(node => {
            if (node.is_visible) {
                let delay = time_remainder - node.offset;
                // map to [-period/2, period/2]
                delay = ((delay + TIMER_PERIOD*1.5) % TIMER_PERIOD) - TIMER_PERIOD/2.0;
                let dx = delay - mu_delay;
                let gaussian_log_likelihood = - Math.log(Math.sqrt(2 * Math.PI)*sigma) - dx * dx / (2 * sigma * sigma)
                // we had 2 outliers in 100 samples, so
                const uniform_likelihood = Math.log( outliers / TIMER_PERIOD);
                let timer_likelihood = Math.max(gaussian_log_likelihood, uniform_likelihood)
                problogic.push_likelihood(node, timer_likelihood, delay);
            }
        });
    }

    console.log("updating trie after updating likelihoods");
    problogic.run_func_w_timing(problogic.update_trie, [true_root, false, 0, lm_request_queue, pDATA, LM, visibility_threshold, tokenizer]);
    problogic.run_func_w_timing(problogic.calc_posteriors, [true_root]); // determine the normalizing constant
    pDATA = true_root.post_Z;
    
    // Convert queue to include tokenization info
    let new_queue = [];
    for (let [key, priority] of Object.entries(lm_request_queue.queue)) {
        new_queue.push([key, priority, true_root.registry[key].tokenization]);
    }
    socket.send(JSON.stringify({type: 'set_queue', content: new_queue}));
    // send the trie
    // let trie_json = JSON.stringify(true_root, ["likelihood", "children", "letter", "val"]);
    // Use replacer function to only include children when ever_visible_parent is true
    function simplifyNode(node) {
        let simple = {
            likelihood: node.likelihood,
            // children: [],
            children: node.children
                .filter(child => child.ever_visible)
                .map(child => simplifyNode(child)),
            letter: node.letter, 
            val: node.val,
            ever_visible_parent: node.ever_visible_parent
        };
        
        return simple;
    }
    let simplified_trie = simplifyNode(true_root);
    console.log("Simplified trie:", simplified_trie);
    socket.send(JSON.stringify({type: 'set_trie', content: simplified_trie}));
    
    let offsets = [];
    // first pass to determine visibility
    for (let node of Object.values(true_root.registry)) {
        let normalized_posterior = node.post_Z - pDATA;
        node.is_visible = normalized_posterior > visibility_threshold;
        node.ever_visible = node.ever_visible || node.is_visible;
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
        offsets[random_order[i]][0].offset = cum_width + widths[random_order[i]];
        cum_width += 2 * widths[random_order[i]];
    }
    if (Math.abs(cum_width - TIMER_PERIOD) > 1e-5) {
        console.log("nodes:", true_root);
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
            delay = ((delay + TIMER_PERIOD*1.5) % TIMER_PERIOD) - TIMER_PERIOD/2.0;
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
let chart;
onMount(async () => {
    // load the tokenizer
    tokenizer = await AutoTokenizer.from_pretrained("llama/llama", {local_files_only: true});
    console.log("Tokenizer loaded");
    true_root.tokenization = problogic.get_tokenization(true_root.val, tokenizer);
    console.log("Tokenization for root:", true_root.tokenization);

    ctx = canvas_element.getContext('2d');
    // Set canvas size
    FIRST_BOX_HEIGHT = canvas_element.clientHeight;
    canvas_element.width = canvas_element.clientWidth;
    canvas_element.height = FIRST_BOX_HEIGHT;

    // Start animation loop
    draw();

    window.addEventListener('keydown', (event) => {
        if (event.code === 'Space') {
            click(event);
            event.preventDefault();
        }
    });
    window.addEventListener('click', (event) => {
        // click(event);
        // event.preventDefault(); 
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
        socket.send(JSON.stringify({type: 'set_queue', content: [[true_root.val, 0, true_root.tokenization]]}));
    });

    socket.addEventListener('message', async (event) => {
        let start = performance.now();
        let response = JSON.parse(event.data);
        let end = performance.now();
        if (response.type === 'processed') {
            console.log("JSON parsing took", end - start, "ms");
            await new Promise(resolve => setTimeout(resolve, 5));
            LM[response.ftp] = {"probs": response.probs, "cum": response.cum, "stop_prob": response.stop_prob};
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
            // !Incorrect: this overwrites requests that may be outside of the update branch if lm_request_queue is cleared which it is not
            //
            // Convert queue to include tokenization info
            let new_queue = [];
            for (let [key, priority] of Object.entries(lm_request_queue.queue)) {
                new_queue.push([key, priority, true_root.registry[key].tokenization]);
            }
            socket.send(JSON.stringify({type: 'set_queue', content: new_queue}));
            // "click" to draw on getting the first
            if (response.ftp === true_root.val) {
                click(null);
            }
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



<!-- NEW DESIGN -->
<div class="flex flex-col w-screen h-screen gap-4 p-4">
    <div class="topbar flex flex-row items-center justify-between w-full gap-4 px-4 h-24">
        <button class="p-3 border-2 rounded-lg hover:bg-gray-100"
            on:click={() => toggle_settings = !toggle_settings}
        >
            <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
        </button>
        <button class="p-3 border-2 rounded-lg hover:bg-gray-100"
            on:click={() => {
                reset();
            }}
        >
            <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
        </button>
        
        <!-- {target_phrase}
        {best_string_agrees_with_target}
        {uncompleted_suffix} -->
        <div class="flex-grow p-2 border-2 rounded-lg bg-white text-4xl font-medium min-h-[3rem]">
            <span class="text-green-700">{best_string_agrees_with_target}</span><span class="text-black">{uncompleted_suffix}</span>
        </div>

        <button class="px-6 py-3 border-2 rounded-lg hover:bg-gray-100 text-xl">Start Session</button>
        <button class="px-6 py-3 border-2 rounded-lg hover:bg-gray-100 text-xl">Tutorial</button>

        {#if !awaiting_first_gesture}
        <div class="border-l-2 pl-6 flex flex-row gap-6">
            <div class="flex items-center gap-2">
                <span class="text-gray-500 text-xl">Time:</span>
                <span class="font-medium text-2xl">{Math.floor(time_elapsed/60)}:{Math.floor(time_elapsed%60).toString().padStart(2,'0')}</span>
            </div>
            <div class="flex items-center gap-2">
                <span class="text-gray-500 text-xl">Words:</span>
                <span class="font-medium text-2xl">{n_words}</span>
            </div>
            <div class="flex items-center gap-2">
                <span class="text-gray-500 text-xl">WPM:</span>
                <span class="font-medium text-2xl">{wpm.toFixed(1)}</span>
            </div>
        </div>
        {:else}
        <div class="border-l-2 pl-6 flex flex-row gap-6">
            <div class="flex items-center gap-2">
                <span class="text-gray-500 text-xl">Time will start when you begin.</span>
            </div>
        </div>
        {/if}

        <div class="border-l-2 pl-6">
            <span class="text-2xl">Hello, Oscar</span>
        </div>
    </div>
    <div class="flex flex-row p-4 gap-4 h-[calc(100%-7rem)]">
        <div class="sidebar flex flex-col w-1/4 flex-shrink-0 border-black border-2 rounded-md p-4" class:hidden={!toggle_settings}>
            <div class="gesture-type flex flex-row w-full space-x-2">
                {#each ['Space', 'Click', 'Blink'] as gesture}
                    <button class="button flex flex-col border-2 border-black rounded-md flex-grow h-16 flex justify-center items-center text-center hover:bg-gray-100 active:bg-gray-200 transition-colors duration-150 shadow-sm hover:shadow-md">{gesture}</button>
                {/each}
            </div>
            <div class="flex flex-col p-4">
                <label for="mu-slider" class="text-sm font-medium">
                    μ={mu_delay_ms}ms
                    {#if !isNaN(empirical_delay)}
                        <span class="text-gray-400 italic">(empirical: {(1000 * empirical_delay).toFixed(1)}ms)</span>
                    {/if}
                </label>
                <input type="range" id="mu-slider" min="0" max="200" bind:value={mu_delay_ms} />
            </div>
            <div class="flex flex-col p-4">
                <label for="sigma-slider" class="text-sm font-medium">
                    σ={sigma_ms}ms
                    {#if !isNaN(empirical_std)}
                        <span class="text-gray-400 italic">(empirical: {(1000 * empirical_std).toFixed(1)}ms)</span>
                    {/if}
                </label>
                <input type="range" id="sigma-slider" min="0" max="150" bind:value={sigma_ms} />
            </div>
            <div class="flex flex-col p-4">
                <label for="outliers-slider" class="text-sm font-medium">
                    Outliers={outliers_perc}%
                    {#if !isNaN(empirical_outliers_perc)}
                        <span class="text-gray-400 italic">(empirical: {empirical_outliers_perc}%)</span>
                    {/if}
                </label>
                <input type="range" id="outliers-slider" min="1" max="15" step="0.1" bind:value={outliers_perc} />
            </div>
            <div class="flex flex-col p-4">
                <label for="timer-period-slider" class="text-sm font-medium">Timer Period={TIMER_PERIOD}s</label>
                <input type="range" id="timer-period-slider" min="0.5" max="2.0" step="0.1" bind:value={TIMER_PERIOD} />
            </div>
            {#key toggle_settings}
            <div bind:this={chart} class="w-full h-[500px] relative border-2 border-black padding-4 overflow-y-auto overflow-x-hidden">
                {#if bin_probs_chart_data && bin_probs_chart_data.length > 0}
                    {#each bin_probs_chart_data as point}
                        <div 
                            class="absolute w-1 h-1 bg-blue-500" 
                            style="left: {point.x * chart.clientWidth}px; bottom: {point.y * 500}px;"
                        />
                    {/each}
                {/if}
                {#if offsets_chart_data && offsets_chart_data.length > 0}
                    {#each offsets_chart_data as point}
                        <div 
                            class="absolute transform rotate-90 whitespace-nowrap origin-left"
                            style="left: {point.x * (chart.clientWidth - 2) + 1}px; top: 500px;"
                        >
                            {point.text}
                        </div>
                        <div
                            class="absolute w-[2px] h-[500px] bg-red-500"
                            style="left: {point.x * (chart.clientWidth - 2) + 1}px; top: 0px;"
                        />
                    {/each}
                {/if}
            </div>
            {/key}
        </div>
        <canvas bind:this={canvas_element} class="h-full w-full bg-black rounded-lg"></canvas>
    </div>
</div>

<Eye on:blink={handleBlink} />