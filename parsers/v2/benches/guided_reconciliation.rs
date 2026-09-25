// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CPU-only benchmark. Arguments: call count, chunk bytes (0 = whole), samples.
//! Measures parser pushes/finish; constructs inputs and checks results outside timing.
use std::{collections::BTreeMap, hint::black_box, time::Instant};

use dynamo_parsers_v2::{
    InvalidGuidedPayloadPolicy, UnifiedParserEvent, UnifiedParserExt, UnifiedParserInit,
    UnifiedToolOutputMode, create_unified_parser_for_family,
};
use serde_json::json;

fn main() {
    let mut args: Vec<usize> = std::env::args()
        .skip(1)
        .filter(|arg| arg != "--bench" && arg != "--test")
        .map(|arg| arg.parse().expect("expected nonnegative integer"))
        .collect();
    if args.is_empty() {
        args = vec![8, 128, 3];
    }
    assert_eq!(args.len(), 3, "arguments: calls chunk_bytes samples");
    let (calls, chunk_bytes, samples) = (args[0], args[1], args[2]);
    assert!(calls > 0 && samples > 0);
    let arguments = r#"{"city":"Paris","unit":"celsius"}"#;
    let payload = format!(
        "[{}]",
        (0..calls)
            .map(|i| format!(r#"{{"name":"weather_{i}","arguments":{arguments}}}"#))
            .collect::<Vec<_>>()
            .join(",")
    );
    // Leave the array close for a separate timed push: all calls have already
    // streamed, but validation and reconciliation have not happened yet.
    let prefix = &payload[..payload.len() - 1];
    let width = if chunk_bytes == 0 {
        prefix.len()
    } else {
        chunk_bytes
    };
    let chunks: Vec<&str> = prefix
        .as_bytes()
        .chunks(width)
        .map(|chunk| std::str::from_utf8(chunk).unwrap())
        .collect();
    let mut total_ns = Vec::new();
    let mut reconcile_ns = Vec::new();
    for sample in 0..samples + 2 {
        let mut parser = create_unified_parser_for_family("qwen3", &[]).unwrap();
        parser
            .initialize_request(UnifiedParserInit {
                tool_output_mode: UnifiedToolOutputMode::GuidedJson { named_tool: None },
                invalid_guided_payload: InvalidGuidedPayloadPolicy::StreamBestEffort,
                ..Default::default()
            })
            .unwrap();
        let mut events = Vec::new();
        let start = Instant::now();
        for chunk in &chunks {
            events.extend(parser.push(black_box(chunk)).unwrap());
        }
        let reconcile_start = Instant::now();
        events.extend(parser.push(black_box("]")).unwrap());
        events.extend(parser.finish().unwrap().events);
        let reconcile = reconcile_start.elapsed().as_nanos() as u64;
        let total = start.elapsed().as_nanos() as u64;
        // Validate every name, verbatim argument, index and completion. This is
        // deliberately outside timing; a faster parser that drops calls fails.
        let mut actual = BTreeMap::<usize, (String, String, usize)>::new();
        for event in black_box(events) {
            match event {
                UnifiedParserEvent::ToolCall(delta) => {
                    let call = actual.entry(delta.tool_index).or_default();
                    if let Some(name) = delta.name {
                        call.0.push_str(&name);
                    }
                    call.1.push_str(&delta.arguments);
                    call.2 += usize::from(delta.complete);
                }
                other => panic!("unexpected event: {other:?}"),
            }
        }
        assert_eq!(actual.len(), calls);
        for (index, (name, args, completed)) in actual {
            assert_eq!(name, format!("weather_{index}"));
            assert_eq!(args, arguments);
            assert_eq!(completed, 1);
        }
        if sample >= 2 {
            total_ns.push(total);
            reconcile_ns.push(reconcile);
        }
    }
    println!(
        "{}",
        json!({
            "calls": calls, "chunk_bytes": chunk_bytes, "payload_bytes": payload.len(),
            "total_ns": total_ns, "reconcile_ns": reconcile_ns
        })
    );
}
