<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Guided parser CPU benchmarks

`compare_guided.py` compares the current worktree with a Git revision, defaulting
to `origin/main`. It exports the baseline into a temporary directory, adds the
same benchmark source, and builds both with `cargo bench --locked`. Existing
checkouts and refs are untouched. The candidate must contain the
`guided_reconciliation` benchmark target.

```bash
python3 benchmarks/compare_guided.py \
  --baseline origin/main \
  --output /tmp/guided-parser-comparison \
  --cpu 0
```

Choose a CPU allowed by the host's affinity settings, or omit `--cpu`. Use a new
output directory for each run. The script refuses to overwrite earlier results.
The controller needs Python 3.12 or newer, Git, the repository's Rust toolchain,
and `taskset` when selecting a CPU. No GPU, serving process, or model is involved.

The default matrix uses 1, 8, 64, 512, and 4,096 parallel calls. Each call has a
unique function name and the same small argument object. `--calls` and `--chunks`
accept comma-separated integers. Large call counts are scaling stress tests,
not representative claims about typical conversations.

Each sample initializes a fresh Qwen Unified parser with required guided JSON
and `StreamBestEffort`. It feeds the payload prefix, then the final `]` in its
own push. Chunk size 0 means the entire prefix in one push, **not** the batch
`parse_complete` API. Size 128 splits the prefix into 128-byte chunks.

- `total_ns` measures pushes and finish, including collecting emitted events.
- `reconcile_ns` measures the last push and finish, including JSON validation,
  payload boundary checks, and reconciliation. It is not an isolated index lookup.
- Parser creation, initialization, input construction, and result validation are
  outside the timed sections.

The benchmark checks every emitted name, verbatim argument string, index, and
completion count after every sample. Missing or duplicated calls fail the run.
Two warm-up parses precede each group. The default comparison runs three rounds
of five measured samples per version and alternates which version runs first.

The output directory contains raw samples, medians and ranges, compiler/CPU
metadata, baseline and candidate source identities, the candidate patch, the
exact benchmark source, and both executables. The candidate may include local
tracked parser changes; its exact saved patch hash is recorded alongside `HEAD`.
The script rejects modified dependencies/manifests outside the parser and
untracked files other than the benchmark and its two helper files. Commit or
isolate those changes before comparing. Builds use separate target directories for baseline and candidate to prevent
same-version artifact reuse. Executable hashes are recorded in the metadata.

These are parser CPU measurements. They do not measure tokenization, model
inference, network latency, tool execution, or full server throughput. A noisy
host can distort small differences; inspect raw samples and repeat a run before
claiming a small improvement.

## Why the reasoning boundary check matters

A guided response may still contain model-specific reasoning or tool markers.
The parser must distinguish control markers outside JSON strings from literal
marker text inside arguments. Previously, the boundary loop searched the entire
remaining suffix for paired reasoning markers at every candidate byte. A
marker-free 769-byte input offered 590,592 suffix bytes to those searches.

The optimized path checks the current offset directly for paired markers and
keeps the existing dynamic-header resolver for channel-based reasoning. The
native regression checks scan work deterministically and verifies that quoted
markers, escaped quotes, UTF-8 text, and actual trailing markers keep their
meaning. Existing conformance checks guard the complete parser outputs.

The separate completeness check still reparses accumulated JSON on successive
pushes. This benchmark intentionally exposes that remaining cost: a faster
finalization step does not imply that the whole streaming path is linear.
