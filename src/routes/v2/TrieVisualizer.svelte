<script>
    import { onMount } from 'svelte';
    import { get, writable } from 'svelte/store';
    import { tweened } from 'svelte/motion';
    import { linear } from 'svelte/easing';
    import { createEventDispatcher } from 'svelte';
    const dispatch = createEventDispatcher();
    import * as colors from '$lib/colors.js';
    let canvas_element;
    let time_elapsed = 0;
    let fps = 0;
    let lastFrameTime = 0;
    let pause_timer = false;
    let wpm_timer_start = 0;
    let left_offset = writable(0);
    let ctx;
    let animationFrameId;
    let FIRST_BOX_HEIGHT;
    let BOX_WIDTH = 37;
    let BOX_WIDTH_CHILDREN_MULTIPLIER = 1.0;
    let TIMER_CIRCLE_RADIUS = 15;
    let TIMER_CIRCLE_WIDTH = 2;
    let TIMER_FONT_SIZE = 37;
    let TIMER_COLOR = 'color';
    let visible_nodes = [];
    let old_visible_registry = {};
    let keep_phases = false;
    let developer_visualizer = false;
    let show_boxes = true;
    const TWEEN_DURATION = 300;
    const TWEEN_EASING = linear;
    let FPS_SMOOTHING = 0.9;
    export let likelihood_model;
    export let trie;
    export let trie_updated_flag;
    export let use_visual_tutor;
    export let target_phrase;
    trie_updated_flag.subscribe(new_trie_updated_flag => {
        if (new_trie_updated_flag) {
            setLocations(trie, true);
            trie_updated_flag.set(false);
        }
    });

    onMount(async () => {
        ctx = canvas_element.getContext('2d');
        FIRST_BOX_HEIGHT = canvas_element.clientHeight;
        
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
                if (document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
                    click(event);
                    event.preventDefault();
                }
            }
        });
        //
        draw();
    });

    let get_timer_frac = (node, time) => {
        return (time - node.phase + likelihood_model.period) % likelihood_model.period / likelihood_model.period;
    }

    function normal_pdf(x, mean, stddev) {
        return Math.exp(-0.5 * Math.pow((x - mean) / stddev, 2)) / (stddev * Math.sqrt(2 * Math.PI));
    }
    function normal_logpdf(x, mean, stddev) {
        return -0.5 * Math.pow((x - mean) / stddev, 2) - Math.log(stddev * Math.sqrt(2 * Math.PI));
    }

    function logaddexp(a, b) {
        if (a === -Infinity) {
            return b;
        }
        if (b === -Infinity) {
            return a;
        }
        if (a > b) {
            return a + Math.log(1 + Math.exp(b - a));
        } else {
            return b + Math.log(1 + Math.exp(a - b));
        }
    }

    function timer_likelihood(time, phase, likelihood_model) {
        // likelihood model has the fields:
        // - mu_delay
        // - stddev_delay
        // - outliers
        // - period
        let delay = time - phase
        // map to [-period/2, period/2]
        delay = ((delay + likelihood_model.period*1.5) % likelihood_model.period) - likelihood_model.period/2.0;
        const gaussian_log_likelihood = normal_logpdf(delay, likelihood_model.mu_delay, likelihood_model.stddev_delay);
        const uniform_log_likelihood = Math.log( 1 / likelihood_model.period);
        const outlier_prob = Math.log(likelihood_model.outliers);
        const not_outlier_prob = Math.log(1 - likelihood_model.outliers);
        return {
            likelihood: logaddexp(not_outlier_prob + gaussian_log_likelihood, outlier_prob + uniform_log_likelihood),
            // TODO: use a tuple since we don't like dragging this about
            delay_pair: {
                delay,
                period: likelihood_model.period,
            }
        }
    }

    function click(event) {
        let time = event.timeStamp / 1000.0;
        let new_likelihoods = {};
        visible_nodes.filter(node => !(node.go_live_time && node.go_live_time > time)).forEach(node => {
            let node_timer_likelihood = timer_likelihood(time, node.phase, likelihood_model);
            new_likelihoods[node.val] = node_timer_likelihood;
        });
        keep_phases = false;
        dispatch('set_likelihoods', new_likelihoods);
    }

    function setLocations(node, root_call = false,  loc = null, size_height = FIRST_BOX_HEIGHT, size_width = BOX_WIDTH) {
        if (root_call) {
            loc = { x: 0, y: 0 };
            size_width *= 1.5;
            visible_nodes = [];
        }
        visible_nodes.push(node);
        node.location = { ...loc };
        // } else {
        if (node.val in old_visible_registry) {
            node.tweened_location = old_visible_registry[node.val].tweened_location;
            node.tweened_location.set(loc);
            node.tweened_size_height = old_visible_registry[node.val].tweened_size_height;
            node.tweened_size_height.set(size_height);
            node.tweened_size_width = old_visible_registry[node.val].tweened_size_width;
            node.tweened_size_width.set(size_width);
        } else {
            node.tweened_location = tweened(loc, { duration: TWEEN_DURATION, easing: TWEEN_EASING });
            node.tweened_size_height = tweened(size_height, { duration: TWEEN_DURATION, easing: TWEEN_EASING });
            node.tweened_size_width = tweened(size_width, { duration: TWEEN_DURATION, easing: TWEEN_EASING });
        }
        node.size_height = size_height;
        node.size_width = size_width;
        // TODO: use more intelligent phases
        if (node.val in old_visible_registry && keep_phases) {
            node.phase = old_visible_registry[node.val].phase;
        } else {
            node.phase = Math.random() * likelihood_model.period;
        }
        let visible_children = node.children.filter(child => child.is_visible || (keep_phases && child.val in old_visible_registry));
        let visible_children_vals = new Set(visible_children.map(child => child.val));

        if (visible_children.length > 0) {
            const numChildren = visible_children.length;
            const box_width_multiplier = 1 + BOX_WIDTH_CHILDREN_MULTIPLIER*Math.log(numChildren);
            let y_relative_bottom = 0;
            let total_children_post_Z = node.children.reduce((acc, child) => logaddexp(acc, child.post_Z), -Infinity);
            // TODO: make shrunk children have a minimum height so they don't overlap
            // TODO: we set_viztrie doesn't guarantee that retained children are valid (think about this)
            node.children.forEach((child, index) => {
                let child_frac = child.post_Z - total_children_post_Z;
                let child_height = size_height * Math.exp(child_frac);
                if (visible_children_vals.has(child.val)) {
                    // set go_live_time for new children (unless this is a likelihood_update in which case everything goes live immediately)
                    if (!(child.val in old_visible_registry)) {
                        let now = performance.now() / 1000.0;
                        console.log("WARNING: child [" + child.val + "] not in old_visible_registry, although node [" + node.val + "] is in old_visible_registry");
                        if (node.val === '' || !keep_phases) {
                            child.go_live_time = now;
                        } else if (node.go_live_time && now < node.go_live_time) {
                            // inherit go_live_time from parent if parent is not yet live
                            child.go_live_time = node.go_live_time;
                        } else {
                            // wait until parent completes the next cycle
                            let time_remaining_on_node = (node.phase - now);
                            time_remaining_on_node = time_remaining_on_node % likelihood_model.period; // map to [-period, period]
                            time_remaining_on_node = (time_remaining_on_node + likelihood_model.period) % likelihood_model.period; // map to [0, period]
                            let time_since_last_completion = likelihood_model.period - time_remaining_on_node;
                            if (time_since_last_completion < 0.250 || (node.go_live_time && now<node.go_live_time+0.250)) {
                                // allow unlimited expansion in the first 250ms after completion or going live
                                child.go_live_time = now;
                            } else {
                                let time_to_live = time_remaining_on_node + likelihood_model.mu_delay + 2.5*likelihood_model.stddev_delay;
                                let go_live_time = now + time_to_live;
                                child.go_live_time = go_live_time;
                            }
                        }
                    }
                    setLocations(child, false, {
                        x: node.location.x + node.size_width,
                        y: node.location.y + y_relative_bottom
                    }, child_height, box_width_multiplier * BOX_WIDTH);
                }
                y_relative_bottom += child_height;
            });
        }

        if (root_call) {
            old_visible_registry = {};
            visible_nodes.forEach(node => {
                old_visible_registry[node.val] = node;
            });
            if (!keep_phases) {
                keep_phases = true;
            }
        }
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
        const visibleNodes = visible_nodes.filter(node => !(node.go_live_time && node.go_live_time > now));
        
        let longest_target_val_onscreen = '';
        visibleNodes.forEach(node => {
            if (target_phrase.startsWith(node.val) && node.val.length > longest_target_val_onscreen.length) {
                longest_target_val_onscreen = node.val;
            }
        });

        // Draw nodes
        visibleNodes.forEach(node => {
            color = colors.color_from_letter(node.letter);
            color_string = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0.13)`;

            ctx.beginPath();
            ctx.rect(
                get(node.tweened_location).x - get(left_offset) + 3,
                get(node.tweened_location).y + 3,
                get(node.tweened_size_width) - 6,
                get(node.tweened_size_height) - 6
            );
            ctx.strokeStyle = color_string;
            ctx.lineWidth = 2;
            if (show_boxes || developer_visualizer) {
                // ctx.stroke();
                ctx.fillStyle = color_string;
                ctx.fill();
            }
            ctx.closePath();
        });
        
        // Draw connections 
        let visible_nodes_registry = {};
        visible_nodes.forEach(node => {
            visible_nodes_registry[node.val] = node;
        });
        visibleNodes.forEach(node => {
            color = colors.color_from_letter(node.letter);
            color_string = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0.5)`;
            let darken_factor = 1.8;
            const darker_color = `rgba(${Math.floor(color[0] / darken_factor)}, ${Math.floor(color[1] / darken_factor)}, ${Math.floor(color[2] / darken_factor)}, 1.0)`;
            if (node.val.length >= 1) {
                let parent_val = node.val.slice(0, -1);
                let parent_node = visible_nodes_registry[parent_val];
                let startX = get(node.tweened_location).x + get(node.tweened_size_width) - get(left_offset) - TIMER_CIRCLE_RADIUS;
                const startY = get(node.tweened_location).y + get(node.tweened_size_height)/2;
                // move to end of circle
                startX += TIMER_CIRCLE_RADIUS;
                let endX = get(parent_node.tweened_location).x + get(parent_node.tweened_size_width) - get(left_offset) - TIMER_CIRCLE_RADIUS;
                // move to start of circle
                endX += TIMER_CIRCLE_RADIUS;
                const endY = get(parent_node.tweened_location).y + get(parent_node.tweened_size_height)/2;
                
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
            let timer_font_size = TIMER_FONT_SIZE;
            let timer_radius = TIMER_CIRCLE_RADIUS;
            if (use_visual_tutor && node.val === longest_target_val_onscreen) {
                timer_font_size = TIMER_FONT_SIZE * 2.0;
                timer_radius = TIMER_CIRCLE_RADIUS * 2.0;
            }
            const centerX = get(node.tweened_location).x + get(node.tweened_size_width) - get(left_offset) - 1*TIMER_CIRCLE_RADIUS;
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
                ctx.font = `${timer_font_size}px verdana, helvetica, sans-serif`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(node.letter, centerX, centerY);
            }
            // beneath in small text write {hello: world}
            if (developer_visualizer) {
                ctx.font = `${timer_font_size/3}px verdana, helvetica, sans-serif`;
                ctx.fillText(`l:${node.likelihood.toFixed(2)}`, centerX, centerY + 1*timer_font_size/3);
                ctx.fillText(`p:${node.prior.toFixed(2)}`, centerX, centerY + 2*timer_font_size/3);
                ctx.fillText(`z:${node.post_Z.toFixed(2)}`, centerX, centerY + 3*timer_font_size/3);
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
            // ctx.globalCompositeOperation = 'source-over';
        });


        animationFrameId = requestAnimationFrame(draw);
    }

</script>

<div class="flex flex-col h-full relative box-border">
    <canvas bind:this={canvas_element} class="h-full w-full bg-black"></canvas>
    <div class="absolute top-4 right-6 flex gap-8 text-white text-2xl">
        <label class="flex items-center gap-3">
            <input type="checkbox" bind:checked={developer_visualizer} class="w-6 h-6"/>
            Debug
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