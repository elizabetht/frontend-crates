// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Harmony recovery policy: EOF recovery and normal-text suppression (audit B10).
//!
//! Decides what survives as user-visible `normal_text` after the grammar has run:
//! strips Harmony protocol envelopes (commentary/analysis/final/message-call) out of
//! the residual, and drops bare text that never carried a Harmony final/commentary
//! message. Depends only on `harmony_grammar` (the cleanup regexes + special-token
//! recording); the parser state machine calls this after `parse_harmony_snapshot`.

use regex::Captures;

use super::harmony_grammar::{
    analysis_block_cleanup_regex, commentary_block_cleanup_regex, commentary_header_cleanup_regex,
    complete_json_prefix_len, final_block_cleanup_regex, message_call_cleanup_regex,
    next_message_boundary, push_unique, record_special_tokens, special_token_regex,
};

pub(super) fn strip_harmony_protocol_from_normal_text(text: &str, reason: &'static str) -> String {
    let mut stripped = Vec::new();

    let cleaned = commentary_block_cleanup_regex()
        .replace_all(text, |caps: &Captures<'_>| {
            record_special_tokens(&caps[0], &mut stripped);
            let item = match caps.name("name").map(|m| m.as_str()) {
                Some(name) => format!("commentary_tool_call:functions.{name}"),
                None => "commentary_tool_call:missing_recipient".to_string(),
            };
            push_unique(&mut stripped, item);
            ""
        })
        .into_owned();

    let cleaned = commentary_header_cleanup_regex()
        .replace_all(&cleaned, |caps: &Captures<'_>| {
            record_special_tokens(&caps[0], &mut stripped);
            let item = match caps.name("name").map(|m| m.as_str()) {
                Some(name) => format!("commentary_tool_call_without_message:functions.{name}"),
                None => "commentary_tool_call_without_message:missing_recipient".to_string(),
            };
            push_unique(&mut stripped, item);
            ""
        })
        .into_owned();

    let mut analysis_cleaned = String::new();
    let mut cursor = 0;
    while let Some(caps) = analysis_block_cleanup_regex().captures_at(&cleaned, cursor) {
        let matched = caps.get(0).expect("analysis envelope");
        let envelope = matched.as_str();
        let channel_end =
            envelope.find("<|channel|>").expect("analysis channel") + "<|channel|>".len();
        let body_start = envelope
            .find("<|message|>")
            .map_or(envelope.len(), |at| at + "<|message|>".len());
        // The regex can stop at a marker inside an argument string. Recover
        // the full JSON extent before deciding which bytes belong to this
        // unfinished envelope, so its tail cannot leak into visible text.
        let absolute_body_start = matched.start() + body_start;
        let json_end = absolute_body_start
            + complete_json_prefix_len(&cleaned[absolute_body_start..]).unwrap_or(0);
        let candidate_end = matched.end().max(json_end);
        let envelope = &cleaned[matched.start()..candidate_end];
        let boundary = next_message_boundary(&envelope[channel_end..body_start])
            .map(|at| channel_end + at)
            .or_else(|| next_message_boundary(&envelope[body_start..]).map(|at| body_start + at));
        let end = boundary.map_or(candidate_end, |at| matched.start() + at);
        analysis_cleaned.push_str(&cleaned[cursor..matched.start()]);
        record_special_tokens(&cleaned[matched.start()..end], &mut stripped);
        push_unique(&mut stripped, "analysis_envelope".to_string());
        cursor = end;
    }
    analysis_cleaned.push_str(&cleaned[cursor..]);
    let cleaned = analysis_cleaned;

    let cleaned = final_block_cleanup_regex()
        .replace_all(&cleaned, |caps: &Captures<'_>| {
            record_special_tokens(&caps[0], &mut stripped);
            push_unique(&mut stripped, "final_envelope".to_string());
            caps.name("body")
                .map(|m| m.as_str())
                .unwrap_or_default()
                .to_string()
        })
        .into_owned();

    let cleaned = message_call_cleanup_regex()
        .replace_all(&cleaned, |caps: &Captures<'_>| {
            record_special_tokens(&caps[0], &mut stripped);
            push_unique(&mut stripped, "message_call_payload".to_string());
            ""
        })
        .into_owned();

    let cleaned = special_token_regex()
        .replace_all(&cleaned, |caps: &Captures<'_>| {
            push_unique(&mut stripped, format!("special_token:{}", &caps[0]));
            ""
        })
        .into_owned();

    if stripped.is_empty() {
        return text.to_string();
    }

    // Only the protocol envelopes are removed — the surrounding plain text is
    // kept VERBATIM, including the boundary space touching a stripped envelope
    // (e.g. `"I will check the weather. "` before `<|channel|>` keeps its
    // trailing space). The v1 jail passes that text through untouched; trimming
    // here made the stream output lose model-emitted whitespace.
    tracing::warn!(
        family = "harmony",
        reason,
        stripped = ?stripped,
        original_len = text.len(),
        cleaned_len = cleaned.len(),
        "stripped harmony protocol content from normal_text"
    );
    cleaned
}

pub(super) fn normal_text_after_parse_failure(text: &str, reason: &'static str) -> String {
    // No calls were parsed: strip any protocol residue and pass the remaining
    // plain text through VERBATIM. Marker-free text (a model answering in bare
    // prose without Harmony framing, or a whitespace-only response) is the
    // user's content and cannot leak markup by definition — dropping it here
    // used to swallow whole answers (the whole-answer-drop class). The v1 jail passes
    // such text through untouched; the strict v1 batch parser still drops it —
    // that divergence is documented in the batch-via-stream allowlist.
    let cleaned = strip_harmony_protocol_from_normal_text(text, reason);
    if cleaned == text && !text.trim().is_empty() {
        tracing::warn!(
            family = "harmony",
            reason,
            original_len = text.len(),
            "passing through bare text without a Harmony final/commentary message"
        );
    }
    cleaned
}
