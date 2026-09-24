// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Harmony grammar: token/channel recognition and tool-call extraction (audit B10).
//!
//! Pure recognition over Harmony protocol text — the `commentary`/`analysis`/`final`
//! channel regexes, special-token recording, JSON-argument serialization, and the
//! `extract_calls_via_regex` scanner. No EOF-recovery or normal-text cleanup policy
//! lives here (that is `harmony_recovery`); this module is the leaf both the parser
//! state machine and the recovery policy build on.

use std::sync::OnceLock;

use regex::Regex;
use serde_json::Value;

static TOOL_CALL_BLOCK_REGEX: OnceLock<Regex> = OnceLock::new();
static COMMENTARY_BLOCK_CLEANUP_REGEX: OnceLock<Regex> = OnceLock::new();
static COMMENTARY_HEADER_CLEANUP_REGEX: OnceLock<Regex> = OnceLock::new();
static ANALYSIS_BLOCK_CLEANUP_REGEX: OnceLock<Regex> = OnceLock::new();
static FINAL_BLOCK_CLEANUP_REGEX: OnceLock<Regex> = OnceLock::new();
static MESSAGE_CALL_CLEANUP_REGEX: OnceLock<Regex> = OnceLock::new();
static SPECIAL_TOKEN_REGEX: OnceLock<Regex> = OnceLock::new();
static TOOL_CALL_BLOCK_EOF_REGEX: OnceLock<Regex> = OnceLock::new();

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CompleteHarmonyCall {
    pub(super) name: String,
    pub(super) arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct HarmonySnapshot {
    pub(super) calls: Vec<CompleteHarmonyCall>,
    pub(super) normal_text: String,
}

// Match the batch parser's recovery of directed analysis calls as well as
// canonical commentary calls. A recipient and <|call|> are still required.
pub(super) fn tool_call_block_regex() -> &'static Regex {
    TOOL_CALL_BLOCK_REGEX.get_or_init(|| {
        Regex::new(
            r"(?s)(?:<\|start\|>assistant)?<\|channel\|>(?:commentary|analysis) to=functions\.(?P<name>[\w.\-]+).*?<\|message\|>(?P<args>.*?)<\|call\|>",
        )
        .expect("tool call block regex")
    })
}

pub(super) fn tool_call_block_eof_regex() -> &'static Regex {
    TOOL_CALL_BLOCK_EOF_REGEX.get_or_init(|| {
        Regex::new(
            r"(?s)(?:<\|start\|>assistant)?<\|channel\|>(?:commentary|analysis) to=functions\.(?P<name>[\w.\-]+).*?<\|message\|>(?P<args>.*?)(?:<\|call\|>|(?P<eof>\z))",
        )
        .expect("tool call block EOF regex")
    })
}

pub(super) fn commentary_block_cleanup_regex() -> &'static Regex {
    COMMENTARY_BLOCK_CLEANUP_REGEX.get_or_init(|| {
        Regex::new(
            r"(?s)(?:<\|start\|>assistant)?<\|channel\|>commentary(?:\s+to=functions\.(?P<name>[\w.\-]+))?.*?<\|message\|>.*?(?:<\|call\|>|\z)",
        )
        .expect("commentary block cleanup regex")
    })
}

pub(super) fn commentary_header_cleanup_regex() -> &'static Regex {
    COMMENTARY_HEADER_CLEANUP_REGEX.get_or_init(|| {
        Regex::new(
            r"(?s)(?:<\|start\|>assistant)?<\|channel\|>commentary(?:\s+to=functions\.(?P<name>[\w.\-]+))?.*\z",
        )
        .expect("commentary header cleanup regex")
    })
}

pub(super) fn analysis_block_cleanup_regex() -> &'static Regex {
    ANALYSIS_BLOCK_CLEANUP_REGEX.get_or_init(|| {
        Regex::new(
            r"(?s)(?:<\|start\|>assistant)?<\|channel\|>analysis(?:\s+to=functions\.[\w.\-]+.*?)?(?:<\|message\|>.*?(?:<\|call\|>|<\|end\|>|\z)|\z)",
        )
        .expect("analysis block cleanup regex")
    })
}

pub(super) fn final_block_cleanup_regex() -> &'static Regex {
    FINAL_BLOCK_CLEANUP_REGEX.get_or_init(|| {
        Regex::new(
            r"(?s)(?:<\|start\|>assistant)?<\|channel\|>final<\|message\|>(?P<body>.*?)(?:<\|return\|>|<\|end\|>|\z)",
        )
        .expect("final block cleanup regex")
    })
}

pub(super) fn message_call_cleanup_regex() -> &'static Regex {
    MESSAGE_CALL_CLEANUP_REGEX.get_or_init(|| {
        Regex::new(r"(?s)<\|message\|>.*?(?:<\|call\|>|\z)").expect("message call cleanup regex")
    })
}

pub(super) fn special_token_regex() -> &'static Regex {
    SPECIAL_TOKEN_REGEX.get_or_init(|| {
        Regex::new(r"<\|(?:start|channel|constrain|message|call|end|return)\|>")
            .expect("special token cleanup regex")
    })
}

pub(super) fn push_unique(items: &mut Vec<String>, item: String) {
    if !items.iter().any(|existing| existing == &item) {
        items.push(item);
    }
}

pub(super) fn record_special_tokens(text: &str, items: &mut Vec<String>) {
    for matched in special_token_regex().find_iter(text) {
        push_unique(items, format!("special_token:{}", matched.as_str()));
    }
}

pub(super) fn serialize_harmony_arguments(raw_args: &str) -> String {
    let trimmed = raw_args.trim();
    match serde_json::from_str::<Value>(trimmed) {
        Ok(value) => serde_json::to_string(&value).unwrap_or_else(|_| trimmed.to_string()),
        Err(_) => trimmed.to_string(),
    }
}

pub(super) fn args_are_complete_json(raw_args: &str) -> bool {
    serde_json::from_str::<Value>(raw_args.trim()).is_ok()
}

pub(super) fn complete_json_prefix_len(text: &str) -> Option<usize> {
    let mut values = serde_json::Deserializer::from_str(text).into_iter::<Value>();
    match values.next() {
        Some(Ok(_)) => Some(values.byte_offset()),
        _ => None,
    }
}

// Do not let an unfinished envelope borrow a later message's terminator.
// A complete JSON value can contain literal protocol-looking strings, so skip
// that value before looking for the next message boundary.
pub(super) fn next_message_boundary(text: &str) -> Option<usize> {
    let offset = complete_json_prefix_len(text).unwrap_or(0);
    ["<|start|>", "<|channel|>", "<|end|>", "<|return|>"]
        .into_iter()
        .filter_map(|marker| text[offset..].find(marker).map(|at| offset + at))
        .min()
}

pub(super) fn extract_calls_via_regex(
    text: &str,
    allow_eof_recovery: bool,
) -> (Vec<CompleteHarmonyCall>, String) {
    let mut out = Vec::new();
    let mut residual = String::new();
    let mut cursor = 0;
    let regex = if allow_eof_recovery {
        tool_call_block_eof_regex()
    } else {
        tool_call_block_regex()
    };
    while let Some(cap) = regex.captures_at(text, cursor) {
        let matched = cap.get(0).expect("regex match has full span");
        let name_match = cap.name("name").expect("tool name");
        let args_match = cap.name("args").expect("tool arguments");
        let header_boundary = next_message_boundary(&text[name_match.end()..args_match.start()])
            .map(|at| name_match.end() + at);
        let body_boundary =
            next_message_boundary(args_match.as_str()).map(|at| args_match.start() + at);
        if let Some(boundary) = header_boundary.or(body_boundary) {
            // Keep the unfinished envelope for cleanup and resume at the next
            // message, which may contain a valid call of its own.
            residual.push_str(&text[cursor..boundary]);
            cursor = boundary;
            continue;
        }
        residual.push_str(&text[cursor..matched.start()]);
        cursor = matched.end();

        let name = cap.name("name").map(|x| x.as_str()).unwrap_or("");
        let raw_args = cap.name("args").map(|x| x.as_str().trim()).unwrap_or("{}");
        if name.is_empty() {
            continue;
        }
        if cap.name("eof").is_some() {
            if !args_are_complete_json(raw_args) {
                continue;
            }
            tracing::warn!(
                family = "harmony",
                reason = "eof_recovered_complete_call_without_call_marker",
                function = name,
                recovered_bytes = raw_args.len(),
                "recovered complete Harmony tool call at EOF"
            );
        }
        out.push(CompleteHarmonyCall {
            name: name.to_string(),
            arguments: serialize_harmony_arguments(raw_args),
        });
    }
    residual.push_str(&text[cursor..]);
    // Residual is returned VERBATIM — the boundary space touching an extracted
    // call span (e.g. the trailing space before the last call's envelope in
    // batch case 8.d) is model text the v1 jail passes through; trimming here
    // silently dropped it before the protocol-strip pass ever saw it.
    (out, residual)
}
