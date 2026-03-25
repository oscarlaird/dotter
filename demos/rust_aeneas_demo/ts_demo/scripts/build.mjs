import { cp, mkdir, rm } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const root = path.resolve(__dirname, "..");
const distDir = path.join(root, "dist");
const wasmSource = path.resolve(
  root,
  "..",
  "target",
  "wasm32-unknown-unknown",
  "release",
  "adder_wasm.wasm",
);
const wasmDest = path.join(distDir, "adder_wasm.wasm");

function run(command, args) {
  const result = spawnSync(command, args, {
    cwd: root,
    stdio: "inherit",
  });

  if (result.status !== 0) {
    throw new Error(`Command failed: ${command} ${args.join(" ")}`);
  }
}

await rm(distDir, { recursive: true, force: true });
await mkdir(distDir, { recursive: true });
await cp(wasmSource, wasmDest);
run("npx", ["tsc", "-p", "tsconfig.json"]);
