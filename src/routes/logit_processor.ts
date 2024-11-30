import * as webllm from "@mlc-ai/web-llm";
// import { asyncLoadTokenizer } from "@mlc-ai/web-llm/lib/cache_util";
import { ChatConfig } from "@mlc-ai/web-llm/lib/config";

import * as tvmjs from "@mlc-ai/web-runtime";

import { MyLogitProcessor } from "./my_logit_processor";


const USE_WEB_WORKER = true; // Toggle this to use Logit Processor without a web worker
const AUTOREGRESS_LIMIT = 32; // How many tokens to generate for this test

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

async function main() {
  // Unlike "Llama-3.1-8B-Instruct-q4f32_1-MLC", this is a base model
  // model info from text-generate example
  // curModelConfig and tokenizer instantiation from engine.ts source code
  // need to expose asyncLoadTokenizer in index.ts of the web-llm package
  // "phi-2-q4f32_1-MLC",
  // const selectedModel = "Llama-3.1-8B-q4f32_1-MLC";
  const selectedModel = "phi-2-q4f32_1-MLC";
  const appConfig: webllm.AppConfig = {
    model_list: [
      {
        // model: "https://huggingface.co/mlc-ai/Llama-3.1-8B-q4f32_1-MLC", // a base model
        model: `https://huggingface.co/mlc-ai/${selectedModel}`,
        model_id: selectedModel,
        model_lib:
          webllm.modelLibURLPrefix +
          webllm.modelVersion +
          // "/Llama-3_1-8B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
          `/${selectedModel}-ctx4k_cs1k-webgpu.wasm`,
        overrides: {
          context_window_size: 2048,
        },
      },
    ],
  };
  const modelUrl = new URL("https://huggingface.co/mlc-ai/Llama-3.1-8B-q4f32_1-MLC/resolve/main/").href; // a base model url
  const configUrl = new URL("mlc-chat-config.json", modelUrl).href;
  let configCache = new tvmjs.ArtifactCache("webllm/config");
  const curModelConfig = {
    ...(await configCache.fetchWithCache(
      configUrl,
      "json",
      // this.reloadController?.signal,
    )),
    // ...modelRecord.overrides,
    // ...chatOpts,
  } as ChatConfig;
  const tokenizer = await webllm.asyncLoadTokenizer(
    modelUrl,
    curModelConfig,
    appConfig,
    // this.logger,
  );
  console.log("tokenizer", tokenizer); 


  const initProgressCallback = (report: webllm.InitProgressReport) => {
    setLabel("model-status-label", report.text);
  };
  // Instantiate myLogitProcessor, registering in the logitProcessorRegistry
  const myLogitProcessor = new MyLogitProcessor();
  const logitProcessorRegistry = new Map<string, webllm.LogitProcessor>();
  // logitProcessorRegistry.set("phi-2-q4f32_1-MLC", myLogitProcessor);
  logitProcessorRegistry.set(selectedModel, myLogitProcessor);
  let engine: webllm.MLCEngineInterface;
  // Depending on whether we use a web worker, the code is slightly different
  if (USE_WEB_WORKER) {
    // see worker.ts on how LogitProcessor plays a role there
    engine = await webllm.CreateWebWorkerMLCEngine(
      new Worker(new URL("./worker.ts", import.meta.url), { type: "module" }),
      // "phi-2-q4f32_1-MLC",
      selectedModel,
      { initProgressCallback: initProgressCallback,
        appConfig: appConfig,
       },
    );
  } else {
    // engine = await webllm.CreateMLCEngine("phi-2-q4f32_1-MLC", {
    engine = await webllm.CreateMLCEngine(selectedModel, {
      initProgressCallback: initProgressCallback,
      appConfig: appConfig,
      logitProcessorRegistry: logitProcessorRegistry,
    });
  }


  // Below we demonstrate the usage of a low-level API `forwardTokensAndSample()`
  const prompt: Array<number> = [42];
  console.log("prompt", prompt, tokenizer.decode(prompt));
  let nextToken = await engine.forwardTokensAndSample(
    prompt,
    /*isPrefill=*/ true,
  );
  console.log(nextToken);

  let counter = prompt.length;
  while (counter < AUTOREGRESS_LIMIT) {
    counter += 1;
    nextToken = await engine.forwardTokensAndSample(
      [nextToken],
      /*isPrefill=*/ false,
    );
    console.log(nextToken, tokenizer.decode([nextToken]));
  }

  // By calling `engine.resetChat()`, we triggers MyLogitProcessor.resetState()
  // engine.resetChat();
  // counter = prompt.length;
  // nextToken = await engine.forwardTokensAndSample(prompt, /*isPrefill=*/ true);
  // console.log(nextToken);
  // while (counter < AUTOREGRESS_LIMIT) {
  //   counter += 1;
  //   nextToken = await engine.forwardTokensAndSample(
  //     [nextToken],
  //     /*isPrefill=*/ false,
  //   );
  //   console.log(nextToken);
  // }

  // `forwardTokensAndSample()` is made compatible with registering runtime stats.
  console.log(await engine.runtimeStatsText());
}

// main();
export { main };
