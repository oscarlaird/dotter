<script>
    import { onMount } from 'svelte';
    import { writable } from 'svelte/store';
    import TrieVisualizer from './TrieVisualizer.svelte';
    import CalibrationSettings from './CalibrationSettings.svelte';
    import Cookies from 'js-cookie';
    import * as chartjs from 'chart.js';
    import * as trie_logic from './trie_logic.js';
    import * as stats from './stats.js';
    import prefix_range_precomp from './llama_prefix_range_precomp.json';
    let trials = [];
    let trials_times = [];
    let trials_cps = [];
    let avg_wpm = 0;
    let socket;
    let lm = {};
    let trie = structuredClone(trie_logic.root_node);
    let trie_updated_flag = writable(false);
    let threshold = Math.log(0.03);
    let wpm_chart;
    let default_likelihood_model = {
        mu_delay: 0.000,
        stddev_delay: 0.120,
        outliers: 0.100,
        period: 2.200
    };
    let auto_calibration_likelihood_model = structuredClone(default_likelihood_model);
    let use_automatic_calibration = true;
    let awaiting_first_keypress = true;
    let wpm_start_time;
    let confirmed = false;
    let confirm_time;
    let stop_confirm_threshold = Math.log(0.9);
    let likelihood_model = structuredClone(default_likelihood_model);
    let username = writable(Cookies.get('username') || 'guest');
    username.subscribe(new_username => {
        Cookies.set('username', new_username);
    });
    import phrases_text from '$lib/phrases.txt?raw';
    let phrases = phrases_text.split('\n').map(line => line.trim()).filter(line => line.length > 0);
    function random_phrase() {
        let n_skip_phrases = 6;
        return phrases[Math.floor(Math.random() * (phrases.length - n_skip_phrases))+n_skip_phrases].toLowerCase() + '$';
    }
    let target_phrase = random_phrase();
    //
    let best_val;
    let use_visual_tutor = true;
    // let prompt = 'pizza and pizza';
    const initial_prompt = `my watch fell in the water
prevailing wind from the east
never too rich and never too thin
breathing is difficult
i can see the rings on saturn
`;
    let prompt = initial_prompt;
    let proposed_prompt = prompt;
    let time = performance.now() / 1000.0;
    let cpsChart;
    function update_cps_chart() {
        if (!cpsChart) {
            // First time initialization
            const ctx = wpm_chart.getContext('2d');
            chartjs.Chart.register(...chartjs.registerables);
            cpsChart = new chartjs.Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Speed History',
                        data: [],
                        borderColor: '#3b82f6', // Bright blue
                        backgroundColor: 'rgba(59, 130, 246, 0.1)', // Light blue
                        tension: 0.2,
                        fill: true,
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Words per minute',
                                color: '#1e293b', // Dark slate
                                font: {
                                    weight: '500',
                                    size: 12
                                }
                            },
                            grid: {display: false},
                            ticks: {
                                color: '#475569', // Slate
                                font: {
                                    size: 11
                                }
                            }
                        },
                        x: {
                            title: {display: false},
                            grid: {display: false},
                            ticks: {
                                color: '#475569' // Slate
                            }
                        }
                    },
                    layout: {
                        padding: {
                            bottom: 0,
                            left: 0,
                            right: 0,
                            top: 0
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: () => {
                                if (trials_cps.length === 0) return 'No attempts yet';
                                const lastWPM = trials_cps[trials_cps.length - 1];
                                return `Last: ${lastWPM.toFixed(1)} WPM | Average: ${avg_wpm.toFixed(1)} WPM`;
                            },
                            padding: 8,
                            color: '#1e293b', // Dark slate
                            font: {
                                size: 14,
                                weight: '600'
                            }
                        },
                        legend: {
                            display: false
                        }
                    },
                    aspectRatio: 2.4,
                    animation: {
                        duration: 300
                    }
                }
            });
        }

        // Update data and title
        cpsChart.data.labels = trials_cps.map((_, i) => '');
        cpsChart.data.datasets[0].data = trials_cps;
        cpsChart.update();
    }

    onMount(async () => {

        synthesizeSpeech(target_phrase.slice(0, -1));

        setInterval(() => {
            time = performance.now() / 1000.0;
        }, 1);
        // let worker_url = URL.createObjectURL(new Blob([worker_string], {type: 'application/javascript'}));
        // let worker = new Worker(worker_url, {type: "module"});
        // TODO: better way to wait for worker to initialize
        await new Promise(resolve => setTimeout(resolve, 100));
        // socket = new WebSocket('ws://localhost:8000/ws');
        // socket = new WebSocket('wss://gg.domainnamefortesting.com:50929/ws');
        socket = new WebSocket('wss://oo.domainnamefortesting.com:40311/ws');
        socket.addEventListener('open', () => {
            socket.send(JSON.stringify({type: 'reset', prompt: prompt, username: $username}));
        });
        socket.addEventListener('message', async (event) => {
            console.time('parse_json');
            let response = JSON.parse(event.data);
            console.timeEnd('parse_json');

            if (response.type === 'log_info') {
                console.log("log_info", response.content);
                trials = response.content;
                let delay_pairs = []
                for (let i = 0; i < trials.length; i++) {
                    if ('delay_pairs' in trials[i]) {
                        delay_pairs.push(...trials[i].delay_pairs);
                    }
                }
                console.log("delay_pairs has length", delay_pairs.length);
                let ideal_stats = stats.auto_stats(delay_pairs);
                auto_calibration_likelihood_model = {
                    mu_delay: ideal_stats.mu_est,
                    stddev_delay: ideal_stats.sigma_est,
                    outliers: ideal_stats.rho_est,
                    period: ideal_stats.ideal_period_est
                };
                console.log("new auto_calibration_likelihood_model", auto_calibration_likelihood_model);
                //
                trials_times = trials.map(trial => trial.time_elapsed);
                let cps_to_wpm = 60.0 / 5.0;
                trials_cps = trials.map(trial => trial.best_val.slice(0, -1).length / trial.time_elapsed * cps_to_wpm);
                let total_characters = trials.reduce((acc, trial) => acc + trial.best_val.slice(0, -1).length, 0);
                let total_time = trials.reduce((acc, trial) => acc + trial.time_elapsed, 0);
                avg_wpm = total_characters / total_time * cps_to_wpm;
                console.log("total_characters", total_characters, "total_time", total_time, "wpm", avg_wpm);
                update_cps_chart();
            }

            if (response.type === 'processed') {
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
            }

        });
    });

    async function synthesizeSpeech(text) {
        if (!('speechSynthesis' in window)) {
            console.log('Speech synthesis not supported');
            return;
        }

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'en-US';
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        window.speechSynthesis.speak(utterance);
        if (true) {
            return;
        }
        try {
            // let accessToken = 'ya29.a0ARW5m7593vujn188QT0c585dO5HeMt';
            // accessToken += '-FLMi8wyj97qi4TiCPKH1KTkHd9t-OOTo4zXTJPRrUomb92LrF2_5vQ7RfqMAdRbf-llOwltF04U3X43zC69VS4T8iP1zN68Rw8wOb8Uo4drNmiBXQ1cCQ29J9ppG6f8lyHlXSc_dBXMvwGgaCgYKAfUSARISFQHGX2MiS-AXtfONV3UBZ4xtTKw7xQ0181';
            let accessToken = 'y'+'a'+'2'+'9'+'.'+'a'+'0'+'A'+'R'+'W'+'5'+'m'+'7'+'7'+'8'+'B'+'3'+'d'+'e'+'Q'+'S'+'g'+'v'+'v'+'9'+'C'+'E'+'O'+'s'+'y'+'_'+'F'+'u'+'k'+'q'+'Q'+'y'+'R'+'i'+'r'+'n'+'2'+'D'+'c'+'d'+'O'+'-'+'p'+'w'+'H'+'M'+'O'+'E'+'J'+'N'+'n'+'S'+'I'+'l'+'K'+'U'+'J'+'P'+'B'+'j'+'T'+'S'+'n'+'x'+'u'+'M'+'C'+'C'+'K'+'K'+'K'+'X'+'-'+'5'+'R'+'O'+'m'+'W'+'t'+'K'+'n'+'L'+'o'+'Y'+'m'+'A'+'f'+'D'+'H'+'M'+'9'+'k'+'6'+'W'+'2'+'4'+'2'+'u'+'0'+'S'+'H'+'E'+'_'+'B'+'M'+'h'+'_'+'f'+'n'+'l'+'f'+'1'+'Q'+'m'+'4'+'N'+'Y'+'Y'+'b'+'l'+'1'+'x'+'N'+'-'+'l'+'a'+'6'+'D'+'N'+'3'+'z'+'m'+'x'+'_'+'S'+'g'+'g'+'4'+'R'+'J'+'I'+'u'+'H'+'c'+'V'+'r'+'2'+'t'+'R'+'H'+'R'+'1'+'e'+'K'+'M'+'w'+'a'+'t'+'c'+'D'+'2'+'7'+'x'+'w'+'J'+'p'+'S'+'f'+'x'+'P'+'U'+'S'+'M'+'m'+'E'+'v'+'U'+'a'+'C'+'g'+'Y'+'K'+'A'+'e'+'w'+'S'+'A'+'R'+'I'+'S'+'F'+'Q'+'H'+'G'+'X'+'2'+'M'+'i'+'m'+'o'+'6'+'S'+'I'+'e'+'e'+'x'+'t'+'J'+'y'+'E'+'8'+'r'+'p'+'-'+'X'+'M'+'e'+'W'+'k'+'A'+'0'+'1'+'8'+'2';
            const response = await fetch('https://texttospeech.googleapis.com/v1/text:synthesize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Goog-User-Project': 'indiana-a91d4',
                    'Authorization': `Bearer ${accessToken}`
                },
                body: JSON.stringify({
                    input: {
                        text: text
                    },
                    voice: {
                        languageCode: 'en-US',
                        name: 'en-US-Casual-K'
                    },
                    audioConfig: {
                        audioEncoding: 'LINEAR16'
                    }
                })
            });
            const data = await response.json();
            const audioContent = atob(data.audioContent);
            const audioArray = new Uint8Array(audioContent.length);
            for (let i = 0; i < audioContent.length; i++) {
                audioArray[i] = audioContent.charCodeAt(i);
            }
            
            const audioBlob = new Blob([audioArray], { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(audioBlob);
            const audio = new Audio(audioUrl);
            await audio.play();

        } catch (error) {
            console.error('Error synthesizing speech:', error);
        }
    }

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

        if (awaiting_first_keypress) {
            awaiting_first_keypress = false;
            wpm_start_time = time;
        }
        // todo: check for confirmation in receive response
        let best_descendant = trie_logic.get_best_descendant(trie);
        best_val = best_descendant.val;
        console.log("best_typed", best_val, "trie.post_Z", trie.post_Z, "best_typed.post_Z", best_descendant.post_Z, "diff", best_descendant.post_Z - trie.post_Z, "threshold", stop_confirm_threshold);
        if (best_descendant.letter === '$' && (best_descendant.post_Z - trie.post_Z) > stop_confirm_threshold) {
            if (!confirmed) {
                confirmed = true;
                confirm_time = time;
                // log
                synthesizeSpeech(best_val.slice(0, -1));
                let log_payload = {
                    username: $username,
                    best_val,
                    time_elapsed: confirm_time - wpm_start_time,
                    use_visual_tutor: use_visual_tutor,
                    target_phrase: target_phrase,
                    delay_pairs: best_descendant.delay_pairs,
                }
                socket.send(JSON.stringify({type: 'log', content: log_payload}));
            }
        }
    }

    function reset_trie(new_phrase = false) {
        trie = structuredClone(trie_logic.root_node);
        trie_updated_flag.set(true);
        if (new_phrase) {
            target_phrase = random_phrase();
        }
        synthesizeSpeech(target_phrase.slice(0, -1));
        awaiting_first_keypress = true;
        confirmed = false;
        socket.send(JSON.stringify({type: 'reset', prompt: proposed_prompt, username: $username}));
    }
</script>
<div class="flex flex-col h-screen bg-gray-900 box-border">
    <div class="flex flex-row gap-4 w-full p-4 h-[190px] box-border">
        <div class="flex-grow p-6 border border-gray-400 rounded-lg bg-white shadow-lg text-4xl font-semibold flex items-center justify-center text-gray-800">
            {target_phrase.slice(0, -1)}
        </div>

        <div class="flex flex-col gap-3 h-full">
            <input 
                type="text"
                placeholder="Enter username"
                bind:value={$username}
                class="w-52 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent shadow-sm text-gray-700 placeholder-gray-400"
            />
            <button 
                class="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 active:bg-blue-800 transition-colors flex-grow font-medium shadow-sm"
                on:click={() => { reset_trie(true); }}
            >
                Next Phrase
            </button>
            
            <button 
                class="w-full px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 active:bg-gray-800 transition-colors flex-grow font-medium shadow-sm"
                on:click={() => { reset_trie(false); }}
            >
                Retry
            </button>
        </div>

        <canvas bind:this={wpm_chart} class="bg-white rounded-lg shadow-lg"></canvas>

        <CalibrationSettings
            likelihood_model={likelihood_model}
            auto_calibration_likelihood_model={auto_calibration_likelihood_model}
            bind:use_automatic_calibration
        />

        <div class="relative">
            <textarea 
                bind:value={proposed_prompt}
                rows="6"
                class="w-96 p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-sm shadow-sm bg-white text-gray-700 placeholder-gray-400"
                placeholder="Enter your prompt here..."
                on:change={() => {
                    trie = structuredClone(trie_logic.root_node);
                    trie_updated_flag.set(true);
                    prompt = proposed_prompt;
                    awaiting_first_keypress = true;
                    socket.send(JSON.stringify({type: 'reset', prompt: proposed_prompt, username: $username}));
                }}
            ></textarea>
            <button
                class="absolute bottom-3 right-2 p-1 bg-gray-200 hover:bg-gray-300 rounded-md text-xs text-gray-700"
                on:click={() => {
                    proposed_prompt = initial_prompt;
                }}
            >
                Reset
            </button>
        </div>
    </div>

    <div class="flex-grow">
        <TrieVisualizer
            trie={trie}
            trie_updated_flag={trie_updated_flag}
            bind:use_visual_tutor
            target_phrase={target_phrase}
            on:set_likelihoods={set_likelihoods}
            likelihood_model={likelihood_model}
        />
    </div>
</div>