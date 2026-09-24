// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mistral v11 generation: `[TOOL_CALLS]name[ARGS]{...}`, repeated per call.
//! Also accepts the deployed `name{...}` spelling. Legacy arrays stay in the
//! generic JSON parser. History-only `[CALL_ID]` is not a generation marker.
//! Spec: https://github.com/mistralai/mistral-common/blob/f3bb6e8be220e44ea37b82467abfac6ac16925f1/src/mistral_common/guidance/grammar_factory.py#L86

use serde_json::value::RawValue;
use uuid::Uuid;

use super::response::{CalledFunction, ToolCallResponse, ToolCallType};

const OPEN: &str = "[TOOL_CALLS]";
const ARGS: &str = "[ARGS]";

pub(crate) fn is_name_format(text: &str) -> bool {
    let Some((_, body)) = text.split_once(OPEN) else {
        return false;
    };
    let body = body.trim_start();
    !body.starts_with(['[', '{']) || body.starts_with(ARGS)
}

/// Parse one complete call. JSON deserialization supplies the byte boundary,
/// so braces, escaped quotes and marker-looking text inside strings are data.
fn call(text: &str) -> Option<(&str, &RawValue, usize)> {
    let body = text.strip_prefix(OPEN)?.trim_start();
    let name_end = body.find(['[', '{'])?;
    let name = body[..name_end].trim_end();
    if name.is_empty()
        || name
            .chars()
            .any(|c| c.is_whitespace() || matches!(c, '}' | ']'))
    {
        return None;
    }
    let args = body[name_end..]
        .strip_prefix(ARGS)
        .unwrap_or(&body[name_end..])
        .trim_start();
    if !args.starts_with('{') {
        return None;
    }
    let mut json = serde_json::Deserializer::from_str(args).into_iter::<&RawValue>();
    let raw = json.next()?.ok()?;
    Some((name, raw, text.len() - args.len() + json.byte_offset()))
}

/// The jail releases one complete call and feeds the remainder through its
/// marker matcher again. Never search for a bracket inside the JSON payload.
pub(crate) fn first_call_end(text: &str) -> Option<usize> {
    let start = text.find(OPEN)?;
    call(&text[start..]).map(|(_, _, end)| start + end)
}

pub(crate) fn parse(text: &str, finalize: bool) -> (Vec<ToolCallResponse>, Option<String>) {
    let mut calls = Vec::new();
    let mut normal = String::new();
    let mut remaining = text;
    while let Some(start) = remaining.find(OPEN) {
        normal.push_str(&remaining[..start]);
        let candidate = &remaining[start..];
        let Some((name, args, end)) = call(candidate) else {
            // A malformed or truncated call has no trustworthy end boundary.
            // Keep completed preceding calls; do not reinterpret a marker in an
            // unfinished JSON string as another call or expose protocol text.
            if finalize {
                tracing::warn!(
                    why = "mistral_incomplete_or_invalid_call",
                    bytes = candidate.len(),
                    "dropping incomplete Mistral call"
                );
            }
            return (calls, Some(normal));
        };
        calls.push(ToolCallResponse {
            id: format!("call-{}", Uuid::new_v4()),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name: name.to_owned(),
                arguments: args.get().to_owned(),
            },
        });
        remaining = &candidate[end..];
    }
    normal.push_str(remaining);
    (calls, Some(normal))
}
