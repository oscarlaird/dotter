import { mkdir, rm } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const root = path.resolve(__dirname, "..");
const distDir = path.join(root, "dist");
const generatedDir = path.join(root, "build", "generated");
const leanSource = path.join(root, "Add.lean");
const generatedC = path.join(generatedDir, "Add.c");
const wasmOutput = path.join(distDir, "add_u64.wasm");

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: root,
    stdio: "inherit",
    ...options,
  });

  if (result.status !== 0) {
    throw new Error(`Command failed: ${command} ${args.join(" ")}`);
  }
}

function capture(command, args) {
  const result = spawnSync(command, args, {
    cwd: root,
    encoding: "utf8",
  });

  if (result.status !== 0) {
    throw new Error(`Command failed: ${command} ${args.join(" ")}`);
  }

  return result.stdout.trim();
}

await rm(distDir, { recursive: true, force: true });
await mkdir(distDir, { recursive: true });
await mkdir(generatedDir, { recursive: true });

const leanPrefix = capture("lean", ["--print-prefix"]);
const leanInclude = path.join(leanPrefix, "include");

run("lean", ["-c", generatedC, leanSource]);

run("clang", [
  "--target=wasm32-wasi",
  "--sysroot=/usr",
  "-O3",
  "-nostdlib",
  "-ffunction-sections",
  "-fdata-sections",
  "-Wl,--no-entry",
  "-Wl,--export=add_u64",
  "-Wl,--gc-sections",
  `-I${leanInclude}`,
  generatedC,
  "-o",
  wasmOutput,
]);

run("npx", ["tsc", "-p", "tsconfig.json"]);
