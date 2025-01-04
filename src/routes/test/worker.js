// Using a single worker for all background tasks

// Synchronous Update Tasks
// - push likelihoods
// - compute parent_priors
// - compute visibility / set timers
// DRAW
// - expand (+push to lm_requests_queue)

// Background Tasks
// pop from lm_responses_queue
// update priors
// expand (+push to lm_requests_queue)

// Load the tokenizer

console.log("I am a worker!");

try {
    import { env, AutoTokenizer } from "@huggingface/transformers";
    env.allowLocalModels = true;
    tokenizer = await AutoTokenizer.from_pretrained("llama/llama", {local_files_only: true});
    console.log(tokenizer.encode("Hi Earth!"));
} catch (error) {
    console.error("Error initializing tokenizer:", error);
}
