<script>
    import { onMount } from 'svelte';
    import script_string from './test_worker.js?raw';
    import worker_string from './worker.js?raw';
    import worker_3_string from './worker_3.js?raw';

    let text = "Hello, world! ";
    // import init from '$lib/trie_worker_rust/pkg/trie_worker_rust.js';
    const sharedBuffer = new SharedArrayBuffer(1024); // 1024 bytes

    let test_worker_1 = null;
    let test_worker_2 = null;

    let socket;
    onMount(async () => {
        console.log("Hello, mount!");
        console.log(new URL('/trie_worker_rust/pkg/trie_worker_rust.js', import.meta.url));
        const blob = new Blob([script_string], {type: 'application/javascript'});
        const url = URL.createObjectURL(blob);
        // test_worker_1 = new Worker(url, { type: "module" });
        // test_worker_1.onmessage = (event) => {
        //     console.log("Main thread received message from worker 1:", event);
        // };
        // test_worker_1.postMessage("We are the main thread sending a message to the worker 1!");
        // test_worker_1.postMessage({type: "sharedBuffer", sharedBuffer: sharedBuffer});

        // test_worker_2 = new Worker(url, { type: "module" });
        // test_worker_2.onmessage = (event) => {
        //     console.log("Main thread received message from worker 2:", event);
        // };
        // test_worker_2.postMessage("We are the main thread sending a message to the worker 2!");
        // test_worker_2.postMessage({type: "sharedBuffer", sharedBuffer: sharedBuffer});
        let worker_3_url = URL.createObjectURL(new Blob([worker_3_string], {type: 'application/javascript'}));
        let worker_3 = new Worker(worker_3_url, {type: "module"});
        
        // Wait a bit for worker 3 to initialize
        await new Promise(resolve => setTimeout(resolve, 100));

        socket = new WebSocket('ws://localhost:8000/ws');
        socket.addEventListener('open', (event) => {
            console.log('WebSocket connection established');
            socket.send(JSON.stringify({type: 'test'}));
        });
        socket.addEventListener('message', async (event) => {
            let response = JSON.parse(event.data);
            if (response.type === 'test') {
                console.log("received test message:", response.content);
                worker_3.postMessage(response);
            }
        });
    });

    function onclick() {
        console.log("click");
        if (!test_worker_1) {
            console.log("the tokenization worker 1 has not been initialized!")
        } else {
            test_worker_1.postMessage({type: "encode", text: text});
        }
        let text_2 = "Hello, world!" + text;
        if (!test_worker_2) {
            console.log("the tokenization worker 2 has not been initialized!")
        } else {
            test_worker_2.postMessage({type: "encode", text: text_2});
        }

    }
</script>

<input type="text" bind:value={text} />
<br>
<button on:click={onclick}>Submit!</button>