// Shared access to bin/tint_convert_cli, built by drivers/webgpu/tint_cli/build.sh.
// The compiler suites fail when it is missing: skipping would report untested conversions as passing.

import { spawnSync } from "child_process";
import { existsSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

export const REPO_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");

export function findTintCli() {
  for (const name of ["tint_convert_cli", "tint_convert_cli.exe"]) {
    const path = join(REPO_ROOT, "bin", name);
    if (existsSync(path)) {
      return path;
    }
  }
  console.error("bin/tint_convert_cli not found. Build it with ./drivers/webgpu/tint_cli/build.sh");
  process.exit(1);
}

// Returns { wgsl, error, stderr }; wgsl is null when the conversion fails.
export function convertSpirvFile(cli, path) {
  const result = spawnSync(cli, [path], { encoding: "utf-8", timeout: 60000 });
  const stderr = result.stderr || "";
  if (result.status !== 0) {
    return { wgsl: null, error: stderr || String(result.error || `exit ${result.status}`), stderr };
  }
  return { wgsl: result.stdout, error: null, stderr };
}
