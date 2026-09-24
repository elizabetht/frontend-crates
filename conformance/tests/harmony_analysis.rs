// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Directed analysis calls are an existing v1 recovery contract. Exercise the
//! same contract through the public v2 text and token APIs, without a model.
use dynamo_parsers::tool_calling::try_tool_call_parse_aggregate_finalize;
use dynamo_parsers_v2::{
    create_tool_parser_for_family, encode_harmony, tool_calling::traits::ToolParseResult,
};

fn check(result: ToolParseResult, expected: &[(&str, &str)], text: &str) {
    let result = result.coalesce_calls();
    assert_eq!(result.normal_text, text);
    assert_eq!(result.calls.len(), expected.len(), "{result:?}");
    for (index, (call, (name, arguments))) in result.calls.iter().zip(expected).enumerate() {
        assert_eq!(call.tool_index, index);
        assert_eq!(call.name.as_deref(), Some(*name));
        assert_eq!(call.arguments, *arguments);
        assert!(call.complete);
    }
}

#[tokio::test]
async fn directed_analysis_calls_match_batch_for_text_and_tokens() {
    for header in [
        "analysis to=functions.get_weather code",
        "analysis to=functions.get_weather",
        "analysis to=functions.get_weather <|constrain|>json",
        "commentary to=functions.get_weather code",
    ] {
        let input = format!(
            "<|channel|>analysis<|message|>Need weather.<|end|><|channel|>{header}<|message|>{{\"location\":\"München🦀\"}}<|call|><|channel|>final<|message|>Done.<|return|>"
        );
        let expected = [("get_weather", "{\"location\":\"München🦀\"}")];
        let (batch, text) = try_tool_call_parse_aggregate_finalize(&input, Some("harmony"), None)
            .await
            .unwrap();
        assert_eq!(batch.len(), 1, "{header}");
        assert_eq!(batch[0].function.name, expected[0].0);
        assert_eq!(batch[0].function.arguments, expected[0].1);
        assert_eq!(text.as_deref(), Some("Done."));
        for at in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            let mut parser = create_tool_parser_for_family("harmony", &[]).unwrap();
            let mut result = parser.push(&input[..at]).unwrap();
            result.append(parser.push(&input[at..]).unwrap());
            result.append(parser.finish().unwrap());
            check(result, &expected, "Done.");
        }
        let mut parser = create_tool_parser_for_family("harmony", &[]).unwrap();
        let mut result = ToolParseResult::default();
        for (at, ch) in input.char_indices() {
            result.append(parser.push(&input[at..at + ch.len_utf8()]).unwrap());
        }
        result.append(parser.finish().unwrap());
        check(result, &expected, "Done.");
        let tokens = encode_harmony(&input).unwrap();
        for size in [1, 3, tokens.len()] {
            let mut parser = create_tool_parser_for_family("harmony", &[]).unwrap();
            let mut result = ToolParseResult::default();
            for chunk in tokens.chunks(size) {
                result.append(parser.push_tokens(chunk).unwrap());
            }
            result.append(parser.finish().unwrap());
            check(result, &expected, "Done.");
            let again = parser.finish().unwrap();
            assert!(again.calls.is_empty() && again.normal_text.is_empty());
        }
    }
}

#[test]
fn unfenced_directed_analysis_is_suppressed_at_eof() {
    for body in [
        r#"{"location":"NYC"}"#,
        r#"{"location":"NY"#,
        r#"{"text":"<|end|>secret"}"#,
    ] {
        let input = format!("<|channel|>analysis to=functions.get_weather code<|message|>{body}");
        check_chunkings(&input, &[], "");
    }
}

#[test]
fn mixed_channels_emit_each_call_once_after_its_call_marker() {
    let mut parser = create_tool_parser_for_family("harmony", &[]).unwrap();
    let prefix = "<|channel|>analysis to=functions.first<|message|>{\"x\":1}";
    let pending = parser.push(prefix).unwrap();
    assert!(pending.calls.is_empty());
    let mut result = parser.push("<|call|>").unwrap();
    assert_eq!(result.calls.iter().filter(|c| c.complete).count(), 1);
    result.append(
        parser
            .push("<|channel|>analysis<|message|>Think between calls.<|end|>")
            .unwrap(),
    );
    result.append(
        parser
            .push("<|channel|>commentary to=functions.second<|message|>{}<|call|>")
            .unwrap(),
    );
    result.append(
        parser
            .push("<|channel|>final<|message|>Finished.<|return|>")
            .unwrap(),
    );
    result.append(parser.finish().unwrap());
    check(
        result,
        &[("first", "{\"x\":1}"), ("second", "{}")],
        "Finished.",
    );
}

#[tokio::test]
async fn fenced_malformed_arguments_keep_the_existing_batch_contract() {
    // The tool parser interprets the envelope. It does not validate or repair
    // argument JSON when the model explicitly completed the call.
    let input = "<|channel|>analysis to=functions.f<|message|>{\"x\":}<|call|>";
    let (calls, _) = try_tool_call_parse_aggregate_finalize(input, Some("harmony"), None)
        .await
        .unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.arguments, "{\"x\":}");
    let mut parser = create_tool_parser_for_family("harmony", &[]).unwrap();
    check(
        parser.parse_complete(input).unwrap(),
        &[("f", "{\"x\":}")],
        "",
    );
}

fn check_chunkings(input: &str, expected: &[(&str, &str)], text: &str) {
    for at in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
        let mut parser = create_tool_parser_for_family("harmony", &[]).unwrap();
        let mut result = parser.push(&input[..at]).unwrap();
        result.append(parser.push(&input[at..]).unwrap());
        result.append(parser.finish().unwrap());
        check(result, expected, text);
    }
    let mut parser = create_tool_parser_for_family("harmony", &[]).unwrap();
    let mut result = ToolParseResult::default();
    for (at, ch) in input.char_indices() {
        result.append(parser.push(&input[at..at + ch.len_utf8()]).unwrap());
    }
    result.append(parser.finish().unwrap());
    check(result, expected, text);
    let tokens = encode_harmony(input).unwrap();
    for size in [1, 3, tokens.len()] {
        let mut parser = create_tool_parser_for_family("harmony", &[]).unwrap();
        let mut result = ToolParseResult::default();
        for chunk in tokens.chunks(size) {
            result.append(parser.push_tokens(chunk).unwrap());
        }
        result.append(parser.finish().unwrap());
        check(result, expected, text);
    }
}

#[test]
fn unfinished_analysis_cannot_borrow_a_later_messages_call_marker() {
    for boundary in [
        "<|channel|>",
        "<|start|>assistant<|channel|>",
        "<|end|><|channel|>",
    ] {
        for pending in [
            "",
            "<|message|>{}",
            "<|message|>{\"x\":",
            "<|message|>{\"x\":\"unfinished",
        ] {
            let input = format!(
                "<|channel|>analysis to=functions.f{pending}{boundary}commentary to=functions.g<|message|>{{\"ok\":true}}<|call|>"
            );
            check_chunkings(&input, &[("g", "{\"ok\":true}")], "");
        }
    }
}

#[test]
fn unfinished_analysis_preserves_the_next_well_formed_final_message() {
    for boundary in ["<|channel|>", "<|start|>assistant<|channel|>"] {
        for pending in [
            "",
            "<|message|>{}",
            r#"<|message|>{"text":"<|end|>secret"}"#,
        ] {
            let input = format!(
                "<|channel|>analysis to=functions.f{pending}{boundary}final<|message|>Done.<|return|>"
            );
            check_chunkings(&input, &[], "Done.");
        }
    }
}

#[test]
fn message_markers_inside_valid_argument_strings_remain_data() {
    let arguments = r#"{"text":"<|start|>assistant<|channel|>final<|message|>literal<|end|>"}"#;
    let input = format!("<|channel|>analysis to=functions.f<|message|>{arguments}<|call|>");
    check_chunkings(&input, &[("f", arguments)], "");
}
