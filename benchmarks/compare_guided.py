#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare the working tree's guided parser with an immutable Git baseline.

Builds the same Rust benchmark against both versions using their native locked
workspace dependencies. The baseline is exported into a temporary directory;
existing checkouts and refs are never modified. Records raw timing samples and
source/build identities, alternating execution order across rounds.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BENCH = Path("parsers/v2/benches/guided_reconciliation.rs")
STANZA = '\n[[bench]]\nname = "guided_reconciliation"\nharness = false\n'


def command(args, cwd=ROOT, **kwargs):
    return subprocess.check_output(args, cwd=cwd, text=True, **kwargs).strip()


def verify_candidate_sources(root=ROOT):
    # Only the parser diff and the separately saved benchmark source are captured.
    # Reject other build inputs rather than silently producing incomplete evidence.
    helper_files = {"benchmarks/compare_guided.py", "benchmarks/README.md"}
    changed = command(["git", "diff", "--name-only", "-z", "HEAD"], cwd=root).split("\0")
    untracked = command(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=root).split("\0")
    unsupported = {path for path in changed if path and not path.startswith("parsers/v2/") and path not in helper_files}
    unsupported.update(path for path in untracked if path and path not in helper_files and path != str(BENCH))
    if unsupported:
        raise RuntimeError("uncaptured candidate changes; commit or isolate them first: " + ", ".join(sorted(unsupported)))


def build(root, target, destination):
    env = dict(os.environ, CARGO_TARGET_DIR=str(target))
    result = command([
        "cargo", "bench", "--locked", "-p", "dynamo-parsers-v2",
        "--bench", "guided_reconciliation", "--no-run", "--message-format=json",
    ], cwd=root, env=env)
    for line in result.splitlines():
        entry = json.loads(line)
        if entry.get("target", {}).get("name") == "guided_reconciliation" and entry.get("executable"):
            shutil.copy2(entry["executable"], destination)
            return destination
    raise RuntimeError("cargo did not report a benchmark executable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="origin/main")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, default=Path("/tmp/frontend-perf-target"))
    parser.add_argument("--calls", default="1,8,64,512,4096")
    parser.add_argument("--chunks", default="0,128", help="comma-separated bytes; 0 means whole prefix")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--cpu", type=int, help="pin benchmark processes to this Linux CPU")
    args = parser.parse_args()
    counts = [int(n) for n in args.calls.split(",")]
    chunks = [int(n) for n in args.chunks.split(",")]
    if min(counts) < 1 or min(chunks) < 0 or args.samples < 1 or args.rounds < 1:
        parser.error("calls, samples, rounds must be positive; chunk sizes must be nonnegative")
    verify_candidate_sources()
    output = args.output.resolve()
    if output.exists():
        parser.error("output directory already exists; choose a new path to preserve prior results")
    output.mkdir(parents=True)
    baseline = command(["git", "rev-parse", "--verify", args.baseline + "^{commit}"])
    candidate = command(["git", "rev-parse", "HEAD"])
    source = (ROOT / BENCH).read_bytes()
    patch = subprocess.check_output(["git", "diff", "HEAD", "--", "parsers/v2"], cwd=ROOT)
    (output / "candidate.patch").write_bytes(patch)
    (output / "benchmark.rs").write_bytes(source)
    metadata = {
        "baseline_ref": args.baseline, "baseline_sha": baseline, "candidate_head": candidate,
        "candidate_patch_sha256": hashlib.sha256(patch).hexdigest(),
        "benchmark_sha256": hashlib.sha256(source).hexdigest(),
        "rustc": command(["rustc", "-Vv"]), "platform": platform.platform(),
        "cpu": args.cpu, "rounds": args.rounds, "samples_per_round": args.samples,
        "build_env": {key: os.environ.get(key) for key in ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC_WRAPPER"]},
        "cpu_info": Path("/proc/cpuinfo").read_text().split("\n\n")[0] if Path("/proc/cpuinfo").exists() else platform.processor(),
    }
    with tempfile.TemporaryDirectory(prefix="frontend-guided-bench-") as temporary:
        temporary = Path(temporary)
        archive = temporary / "baseline.tar"
        subprocess.run(["git", "archive", "--format=tar", f"--output={archive}", baseline], cwd=ROOT, check=True)
        base = temporary / "baseline"
        base.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(base, filter="data")
        (base / BENCH).parent.mkdir(parents=True, exist_ok=True)
        (base / BENCH).write_bytes(source)
        manifest = base / "parsers/v2/Cargo.toml"
        if 'name = "guided_reconciliation"' not in manifest.read_text():
            with manifest.open("a") as stream:
                stream.write(STANZA)
        metadata["baseline_lock_sha256"] = hashlib.sha256((base / "Cargo.lock").read_bytes()).hexdigest()
        metadata["candidate_lock_sha256"] = hashlib.sha256((ROOT / "Cargo.lock").read_bytes()).hexdigest()
        binaries = {
            "baseline": build(base, args.target_dir.resolve() / "baseline" / baseline, output / "baseline-bench"),
            "candidate": build(ROOT, args.target_dir.resolve() / "candidate", output / "candidate-bench"),
        }
    metadata["binary_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in binaries.items()}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    results = []
    with (output / "samples.jsonl").open("w") as raw:
        for calls in counts:
            for chunk in chunks:
                collected = {name: {"total_ns": [], "reconcile_ns": []} for name in binaries}
                for round_index in range(args.rounds):
                    order = ["baseline", "candidate"] if round_index % 2 == 0 else ["candidate", "baseline"]
                    for name in order:
                        argv = [str(binaries[name]), str(calls), str(chunk), str(args.samples)]
                        if args.cpu is not None:
                            argv = ["taskset", "-c", str(args.cpu), *argv]
                        measurement = json.loads(command(argv))
                        measurement.update(implementation=name, round=round_index)
                        raw.write(json.dumps(measurement) + "\n")
                        raw.flush()
                        for metric in collected[name]:
                            collected[name][metric].extend(measurement[metric])
                row = {"calls": calls, "chunk_bytes": chunk}
                for metric in ["total_ns", "reconcile_ns"]:
                    for name in binaries:
                        values = collected[name][metric]
                        row[f"{name}_{metric}_median"] = statistics.median(values)
                        row[f"{name}_{metric}_min"] = min(values)
                        row[f"{name}_{metric}_max"] = max(values)
                    row[f"{metric}_speedup"] = row[f"baseline_{metric}_median"] / row[f"candidate_{metric}_median"]
                results.append(row)
                print(json.dumps(row), flush=True)
                (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
