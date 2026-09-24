<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3 thinking example fixture

`qwen3-0.6b.jinja` is the unmodified `chat_template` string from
[Qwen/Qwen3-0.6B tokenizer_config.json at c1899de](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer_config.json).
The upstream model and template are provided by the Qwen team under Apache-2.0.
The fixture is kept verbatim so its behavior can be reproduced offline.
Its SHA-256 is `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`.

From the repository root:

```bash
cargo run --locked -p dynamo-renderer --example qwen3_thinking
cargo test --locked -p dynamo-renderer --example qwen3_thinking
```

The example sends the same request and tools through the public HF formatter
with `enable_thinking=true` and `false`. It prints both complete prompts.
The disabled prompt adds `<think>\n\n</think>\n\n` after the assistant header.
The tests compare the complete prompts and check that tool instructions survive.

This template reads `enable_thinking`. The HF formatter does not translate
`reasoning_effort=none` or the `thinking=false` alias into that argument.
The consumer supplies model-specific template arguments through
`OAIChatLikeRequest::chat_template_args`. Other Qwen templates, including the
Thinking-2507 template already tested in the renderer, have different behavior.

This is a request-to-prompt test. It does not prove that a model obeys the prompt,
run constrained decoding, or execute `get_weather`. The ordinary Qwen3 tool
instruction here uses JSON inside `<tool_call>`; the parser crate's Unified
`qwen3` selector uses the different Qwen3-Coder XML grammar.
