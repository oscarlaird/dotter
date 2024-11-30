<script>
// import { env, AutoTokenizer } from "@huggingface/transformers";
import { onMount } from 'svelte';
import { tweened } from 'svelte/motion';
import { linear, cubicOut } from 'svelte/easing';
import { get } from 'svelte/store';
import Eye from './eye.svelte';
// import { main } from './logit_processor';
import * as problogic from '$lib/tokenizer';
let LM = {};
let FIRST_BOX_HEIGHT = 800;
let BOX_WIDTH = 60;
let TIMER_CIRCLE_RADIUS = 14;
let TIMER_PERIOD = 1.4;
let BAR_PERIOD = 1.0; // 2 seconds
let time = 0.0;

$: bar_height = (time/BAR_PERIOD % 1) * FIRST_BOX_HEIGHT;


let true_root = {prior: 0, likelihood: 0, children: [], prior_ill: 0, ever_visible_parent: false};
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

//
function get_best_string(current, node) {
    current += node.letter;
    if (node.children.length === 0) {
        return current;
    }
    // Find child with highest post_Z
    let best_child = node.children[0];
    for (const child of node.children) {
        if (child.post_Z > best_child.post_Z) {
            best_child = child;
        }
    }
    return get_best_string(current, best_child);
}
$: best_string = get_best_string(true_root.val.slice(0, -1), true_root);

//
function handleBlink(event) {
    console.log("Blink event:", event);
    click();
}

// Function to recursively set locations for each node in the trie
// Starting from the root (0,0) and calculating child positions based on angle and distance
let visibility_threshold = Math.log(0.05);
let pDATA = 0;
const TWEEN_DURATION = 300;
const TWEEN_EASING = linear;
function setLocations(node, loc = { x: 0, y: 0 }, size_height = FIRST_BOX_HEIGHT, size_width = BOX_WIDTH) {
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
    // filter for .post_Z > threshold
    let visible_children = node.children.filter(child => child.is_visible);

    if (visible_children.length > 0) {
        node.ever_visible_parent = true;
        const numChildren = visible_children.length;
        const box_width_multiplier = 0.5 + Math.log(numChildren);
        // const small_random_angle = Math.random() * 0.5 - 0.25;
        visible_children.forEach((child, index) => {
            setLocations(child, {
                x: node.location.x + node.size_width,
                y: node.location.y + node.size_height * child.y_relative_bottom,
            }, node.size_height * (child.y_relative_top - child.y_relative_bottom), box_width_multiplier * BOX_WIDTH);
        });
    }

    if (node.height === 0) {
        // refresh
        true_root = true_root;
    }
}
setLocations(true_root);

let get_timer_frac = (node, time) => {
    return (time - node.offset + node.period) % node.period / node.period;
}

let socket;
function click() {
    if (!socket) {
        console.log("No socket; can't click");
        return;
    }
    // push likelihoods

    // // TIMER LIKELIHOODS
    Object.values(true_root.registry).forEach(node => {
        if (node.is_visible) {
            let timerFrac = get_timer_frac(node, time);
            timerFrac = (timerFrac + 0.5) % 1.0; // ideal timing will be shifted to 0.5
            let timer_likelihood = - Math.pow(0.5 - timerFrac, 2) * 40.0;
            problogic.push_likelihood(node, timer_likelihood);
            // set a new random offset
            node.offset = Math.random() * node.period;
        }
    });
    // BAR LIKELIHOODS
    // Object.values(true_root.registry).forEach(node => {
    //     if (node.is_visible) {
    //         let node_center = node.location.y + node.size_height;
    //         let dy = Math.abs(node_center - bar_height) / FIRST_BOX_HEIGHT;
    //         dy = Math.min(dy, 1.0 - dy); // wrap around
    //         let bar_likelihood = - dy * 20.0;
    //         problogic.push_likelihood(node, bar_likelihood);
    //     }
    // });

    // update trie, expanding nodes
    let queue = [];
    // pDATA refers to the previous iteration's P(D[:n])
    // but we want P(D[:n+1])
    // so pDATA is an upper bound on P(D[:n+1])
    // but we want a lower bound, so we use pDATA - 1.2
    problogic.update_trie(true_root, false, 0, queue, pDATA, LM, visibility_threshold);
    // fix post_Z for parent nodes
    // 
    problogic.calc_posteriors(true_root);
    pDATA = true_root.post_Z; // posterior probability of the root node, normalized it should be 1, so this is the normalizing constant
    // send queue
    // queue contains [val, post_ill_Z], sort by post_ill_Z descending
    queue.sort((a, b) => b[1] - a[1]);
    console.log("Queue:", queue);
    socket.send(JSON.stringify({type: 'set_queue', content: queue.map(q => q[0])}));
    // set visibility
    for (let node of Object.values(true_root.registry)) {
        node.is_visible = node.post_Z - pDATA > visibility_threshold;
    }
    // TODO: set timers based on posteriors
    // set locations
    setLocations(true_root);
}

onMount(async () => {
    // env.allowLocalModels = true;
    // const tokenizer = await AutoTokenizer.from_pretrained("llama/llama", {local_files_only: true});
    // console.log("Hello world! Encoded:", tokenizer.encode("Hello, world!"));

    setInterval(() => {
        time += 0.050;
    }, 50);

    window.addEventListener('keydown', (event) => {
        if (event.code === 'Space') {
            click();
            event.preventDefault();
        }
    });

    // Set up WebSocket connection
    socket = new WebSocket('ws://localhost:8000/ws');
    // Connection opened
    socket.addEventListener('open', (event) => {
        console.log('WebSocket connection established');
        socket.send(JSON.stringify({type: 'set_queue', content: [true_root.val]}));
        // wait 200ms before resending
        setTimeout(() => {
            // socket.send(JSON.stringify({type: 'set_queue', content: ['1', '12', '123']}));
            socket.send(JSON.stringify({type: 'echo', content: 'Hello, World!'}));

        }, 200);
    });

    // Listen for messages
    socket.addEventListener('message', async (event) => {
        // console.log('Message from server:', JSON.parse(event.data));
        let response = JSON.parse(event.data);
        if (response.type === 'processed') {
            // wait 5ms to allow the DOM to update
            await new Promise(resolve => setTimeout(resolve, 5));
            // TODO: smarter yield to render
            LM[response.ftp] = {"probs": response.probs, "cum": response.cum};
            true_root.registry[response.ftp].in_character_model = true;
            let queue = [];

            // problogic.update_trie(true_root.registry[response.ftp], true, 0, queue, pDATA, LM, visibility_threshold);
            problogic.run_func_w_timing(problogic.update_trie, [true_root.registry[response.ftp], true, 0, queue, pDATA, LM, visibility_threshold]);
            // console.log("Done updating trie");
            // calculate posteriors
            // problogic.calc_posteriors(true_root);
            problogic.run_func_w_timing(problogic.calc_posteriors, [true_root]);
            pDATA = true_root.post_Z; // posterior probability of the root node, normalized it should be 1, so this is the normalizing constant
            // redraw trie
            setLocations(true_root);
            // for (let i = 0; i < true_root.registry.length; i++) {
                // console.log(problogic.node_to_string(true_root.registry[i]));
            // }
            // queue contains [val, post_ill_Z], sort by post_ill_Z descending
            queue.sort((a, b) => b[1] - a[1]);
            console.log("--------------------------------");
            console.log("FTP:", response.ftp);
            console.log("ROOT:", true_root);
            console.log("Queue:", queue);
            console.log("--------------------------------");
            socket.send(JSON.stringify({type: 'set_queue', content: queue.map(q => q[0])}));

        }
    });

    // Connection closed
    socket.addEventListener('close', (event) => {
        console.log('WebSocket connection closed');
    });

    // Connection error
    socket.addEventListener('error', (event) => {
        console.error('WebSocket error:', event);
    });
});
</script>

<Eye on:blink={handleBlink} />
<div class="text-2xl font-bold">
    {best_string}
</div>
<svg viewBox="0 0 6000 6000" class="w-[6000px] h-[6000px]">
    {#key time}
    {#each Object.values(true_root.registry).filter(node => node.is_visible) as node}
        <rect
            x={get(node.tweened_location).x}
            y={get(node.tweened_location).y}
            width={get(node.tweened_size_width)}
            height={get(node.tweened_size_height)}
            fill="lightblue"
            stroke="black"
            stroke-width="2"
        />
    {/each}
    {#each Object.values(true_root.registry).filter(node => node.is_visible) as node}
        {#if node.parent}
            <line
                x1={get(node.tweened_location).x + get(node.tweened_size_width)/2}
                y1={get(node.tweened_location).y + get(node.tweened_size_height)/2} 
                x2={get(node.parent.tweened_location).x + get(node.parent.tweened_size_width)/2}
                y2={get(node.parent.tweened_location).y + get(node.parent.tweened_size_height)/2}
                stroke="black"
                stroke-width="2"
            />
        {/if}
    {/each}
    {#each Object.values(true_root.registry).filter(node => node.is_visible) as node}
        <circle 
            cx={get(node.tweened_location).x + get(node.tweened_size_width)/2} 
            cy={get(node.tweened_location).y + get(node.tweened_size_height)/2} 
            r={TIMER_CIRCLE_RADIUS} 
            fill="lightblue" 
            stroke="black" 
            stroke-width="2"
            z-index="10"
        />
        <path
            d={`M ${get(node.tweened_location).x + get(node.tweened_size_width)/2 + TIMER_CIRCLE_RADIUS} ${get(node.tweened_location).y + get(node.tweened_size_height)/2} A ${TIMER_CIRCLE_RADIUS} ${TIMER_CIRCLE_RADIUS} 0 ${get_timer_frac(node, time) > 0.5 ? 1 : 0} 
             1 ${get(node.tweened_location).x + get(node.tweened_size_width)/2 + TIMER_CIRCLE_RADIUS * Math.cos(2 * Math.PI * get_timer_frac(node, time))} ${get(node.tweened_location).y + get(node.tweened_size_height)/2 + TIMER_CIRCLE_RADIUS * Math.sin(2 * Math.PI * get_timer_frac(node, time))}`}
            fill="none"
            stroke="red"
            stroke-width="4"
            z-index="1"
        />
        <text
            x={get(node.tweened_location).x + get(node.tweened_size_width)/2}
            y={get(node.tweened_location).y + get(node.tweened_size_height)/2}
            text-anchor="middle"
            dominant-baseline="central"
            font-size="20"
            fill="black"
        >
            {node.letter}
        </text>
        <!-- <text
            x={get(node.tweened_location).x + get(node.tweened_size_width)/2}
            y={get(node.tweened_location).y + get(node.tweened_size_height)/2 + 2*TIMER_CIRCLE_RADIUS}
            text-anchor="middle"
            dominant-baseline="central"
            font-family="monospace"
            font-size="12"
            fill="black"
            white-space="pre"
        >
L: {node.likelihood.toFixed(2)}
-: {node.prior.toFixed(2)}
i: {node.prior_ill.toFixed(2)}
+: {(node.post_Z - pDATA)?.toFixed(2) || "?.??"}
        </text> -->
    {/each}
    <!-- <line
        x1="0"
        y1={bar_height}
        x2="6000"
        y2={bar_height}
        stroke="red"
        stroke-width="3"
    /> -->
    {/key}
</svg>
