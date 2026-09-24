// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers_v2::{
    UnifiedEvent, UnifiedParserEvent, UnifiedParserExt, assemble, create_unified_parser_for_family,
    unified::{InvalidGuidedPayloadPolicy, UnifiedParserInit, UnifiedToolOutputMode},
};

fn check(family: &str, chunks: &[&str], invalid: &str, first_index: usize, second_index: usize) {
    let mut parser = create_unified_parser_for_family(family, &[]).unwrap();
    parser
        .initialize_request(UnifiedParserInit {
            tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
            invalid_guided_payload: InvalidGuidedPayloadPolicy::StreamBestEffort,
            ..Default::default()
        })
        .unwrap();
    let mut events = Vec::new();
    for chunk in chunks {
        events.extend(parser.push(chunk).unwrap());
    }
    events.extend(parser.finish().unwrap().events);
    let names: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            UnifiedParserEvent::ToolCall(delta) => {
                delta.name.as_deref().map(|name| (delta.tool_index, name))
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        names,
        [(first_index, "f"), (second_index, "g")],
        "{chunks:?}: {events:?}"
    );
    let completions: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            UnifiedParserEvent::ToolCall(delta) if delta.complete => Some(delta.tool_index),
            _ => None,
        })
        .collect();
    assert_eq!(completions, [first_index, second_index], "{chunks:?}");
    let assembled = assemble(&events);
    let calls: Vec<_> = assembled
        .iter()
        .filter_map(|event| match event {
            UnifiedEvent::ToolCall { name, arguments } => Some((name.as_str(), arguments.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(
        calls,
        [
            ("f", serde_json::json!({"city": "東京, }"})),
            ("g", serde_json::json!({}))
        ],
        "{chunks:?}"
    );
    let recovery: String = assembled
        .iter()
        .filter_map(|event| match event {
            UnifiedEvent::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(recovery, invalid, "{chunks:?}");
}

#[test]
fn invalid_array_elements_keep_streamed_and_completed_call_indices_aligned() {
    // The conformance Init currently selects RecoverAsText. This regression
    // needs the public opt-in StreamBestEffort policy and checks wire indices,
    // not just assembled calls, because orphan provisional deltas are a bug too.
    for &family in dynamo_parsers_v2::builtin_unified_families() {
        for invalid in [
            "null",
            "true",
            "false",
            "42",
            "-1.25e+2",
            r#""東京, }""#,
            "[]",
            r#"[null,{"nested":[1,2]}]"#,
            "{}",
        ] {
            for middle in [false, true] {
                let f = r#"{"name":"f","arguments":{"city":"東京, }"}}"#;
                let g = r#"{"name":"g","arguments":{}}"#;
                let input = if middle {
                    format!("[{f},{invalid},{g}]")
                } else {
                    format!("[{invalid},{f},{g}]")
                };
                let first_index = usize::from(!middle);
                check(family, &[&input], invalid, first_index, 2);
                // Every legal two-chunk split, including name, JSON, and UTF-8
                // boundaries. &str input cannot split inside a UTF-8 code point.
                for at in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
                    check(
                        family,
                        &[&input[..at], &input[at..]],
                        invalid,
                        first_index,
                        2,
                    );
                }
                let tiny: Vec<_> = input
                    .char_indices()
                    .map(|(at, ch)| &input[at..at + ch.len_utf8()])
                    .collect();
                check(family, &tiny, invalid, first_index, 2);
            }
        }
    }
}

#[test]
fn adjacent_invalid_elements_do_not_delay_provisional_calls_until_array_close() {
    for &family in dynamo_parsers_v2::builtin_unified_families() {
        let mut parser = create_unified_parser_for_family(family, &[]).unwrap();
        parser
            .initialize_request(UnifiedParserInit {
                tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                invalid_guided_payload: InvalidGuidedPayloadPolicy::StreamBestEffort,
                ..Default::default()
            })
            .unwrap();
        // The outer array is still open. A consumer must already have the
        // function name and complete argument bytes at source position 2.
        let prefix = r#"[null,[],{"name":"f","arguments":{"city":"東京, }"}},false,"#;
        let mut events = parser.push(prefix).unwrap();
        let partial: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                UnifiedParserEvent::ToolCall(d) => Some(d),
                _ => None,
            })
            .collect();
        assert!(!partial.is_empty(), "{family}: buffered until array close");
        assert!(
            partial.iter().all(|d| d.tool_index == 2 && !d.complete),
            "{family}: {partial:?}"
        );
        assert_eq!(
            partial
                .iter()
                .filter_map(|d| d.name.as_deref())
                .collect::<Vec<_>>(),
            ["f"]
        );
        assert_eq!(
            partial
                .iter()
                .map(|d| d.arguments.as_str())
                .collect::<String>(),
            r#"{"city":"東京, }"}"#
        );
        events.extend(parser.push(r#"{"name":"g","arguments":{}},42]"#).unwrap());
        events.extend(parser.finish().unwrap().events);
        let complete: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                UnifiedParserEvent::ToolCall(d) if d.complete => Some(d.tool_index),
                _ => None,
            })
            .collect();
        assert_eq!(complete, [2, 4], "{family}: {events:?}");
        let assembled = assemble(&events);
        let names: Vec<_> = assembled
            .iter()
            .filter_map(|e| match e {
                UnifiedEvent::ToolCall { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(names, ["f", "g"]);
        let recovery: String = assembled
            .iter()
            .filter_map(|e| match e {
                UnifiedEvent::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(recovery, "null[]false42", "{family}");
    }
}
