<script>
    import { onMount } from 'svelte';
    import { writable } from 'svelte/store';
    import TrieVisualizer from './TrieVisualizer.svelte';
    import CalibrationSettings from './CalibrationSettings.svelte';
    import Cookies from 'js-cookie';
    import * as trie_logic from './trie_logic.js';
    import prefix_range_precomp from './llama_prefix_range_precomp.json';
    let socket;
    let lm = {};
    let trie = structuredClone(trie_logic.root_node);
    let trie_updated_flag = writable(false);
    let threshold = Math.log(0.03);
    let default_likelihood_model = {
        mu_delay: 0.000,
        stddev_delay: 0.120,
        outliers: 0.100,
        period: 2.200
    };
    let likelihood_model = JSON.parse(Cookies.get('likelihood_model') || JSON.stringify(default_likelihood_model));
    let username = writable(Cookies.get('username') || 'guest');
    username.subscribe(new_username => {
        Cookies.set('username', new_username);
    });
    // let prompt = 'pizza and pizza';
    let initial_prompt = `PHRASES:
my watch fell in the water
prevailing wind from the east
never too rich and never too thin
breathing is difficult
i can see the rings on Saturn
`;
    let prompt = initial_prompt;
    let proposed_prompt = prompt;

    onMount(async () => {
        // let worker_url = URL.createObjectURL(new Blob([worker_string], {type: 'application/javascript'}));
        // let worker = new Worker(worker_url, {type: "module"});
        // TODO: better way to wait for worker to initialize
        await new Promise(resolve => setTimeout(resolve, 100));
        socket = new WebSocket('ws://localhost:8000/ws');
        socket.addEventListener('open', () => {
            socket.send(JSON.stringify({type: 'reset', prompt: prompt}));
        });
        socket.addEventListener('message', async (event) => {
            console.time('parse_json');
            let response = JSON.parse(event.data);
            console.timeEnd('parse_json');

            console.time('lm_update');
            let text_after_prompt = response.ftp.slice(prompt.length);
            lm[text_after_prompt] = {"cum": response.cum, "stop_prob": response.stop_prob, "prior_ill": response.prior_ill};
            console.timeEnd('lm_update');

            console.time('update_prior_pipeline');
            trie_logic.update_prior_pipeline(trie, text_after_prompt, lm, prefix_range_precomp);
            console.timeEnd('update_prior_pipeline');

            console.time('set_viztrie_new');
            let pDATA = trie.post_Z;
            trie_logic.set_viztrie_new(trie, lm, threshold, prefix_range_precomp, pDATA);
            trie_updated_flag.set(true);
            console.timeEnd('set_viztrie_new');
        });
    });

    function set_likelihoods(event) {
        if (socket) {
            socket.send(JSON.stringify({type: 'timer_likelihoods', content: event.detail}));
        } else {
            console.log("No socket connection, skipping likelihood set");
        }
        console.time('l pushl_recalc_post_Z_new');
        trie_logic.pushl_recalc_post_Z_new(trie, event.detail);
        console.timeEnd('l pushl_recalc_post_Z_new');

        console.time('l set_viztrie_new');
        let pDATA = trie.post_Z;
        trie_logic.set_viztrie_new(trie, lm, threshold, prefix_range_precomp, pDATA);
        trie_updated_flag.set(true);
        console.timeEnd('l set_viztrie_new');
    }
</script>

<div>
    <textarea bind:value={proposed_prompt} rows="7" cols="50" class="border border-black rounded-md"></textarea>
    <button 
        class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 mr-2"
        on:click={() => {
            trie = structuredClone(trie_logic.root_node);
            trie_updated_flag.set(true);
            prompt = proposed_prompt;
            socket.send(JSON.stringify({type: 'reset', prompt: proposed_prompt}));
        }}>Set Prompt</button>
    <button 
        class="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
        on:click={() => {
            proposed_prompt = initial_prompt;
        }}>Reset Prompt</button>
</div>

<div class="mt-4 mb-4">
    <input 
        type="text" 
        placeholder="Enter username" 
        class="px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        bind:value={$username}
    />
</div>


<CalibrationSettings likelihood_model={likelihood_model}/>
{JSON.stringify(likelihood_model)}
<TrieVisualizer trie={trie} trie_updated_flag={trie_updated_flag} on:set_likelihoods={set_likelihoods} likelihood_model={likelihood_model}/>