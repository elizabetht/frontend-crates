// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers::tool_calling::jail::{Annotated, JailedStream};
use dynamo_parsers::tool_calling::try_tool_call_parse_aggregate_finalize;
use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionStreamResponseDelta,
    CreateChatCompletionStreamResponse, Role,
};
use futures::{StreamExt, stream};

fn chunk(content: impl Into<String>) -> Annotated<CreateChatCompletionStreamResponse> {
    #[allow(deprecated)]
    let choice = ChatChoiceStream {
        index: 0,
        delta: ChatCompletionStreamResponseDelta {
            role: Some(Role::Assistant),
            content: Some(ChatCompletionMessageContent::Text(content.into())),
            tool_calls: None,
            function_call: None,
            refusal: None,
            reasoning_content: None,
        },
        finish_reason: None,
        logprobs: None,
    };
    Annotated {
        data: Some(CreateChatCompletionStreamResponse {
            id: "incremental-jail-test".to_string(),
            choices: vec![choice],
            created: 0,
            model: "test-model".to_string(),
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
            service_tier: None,
        }),
        id: None,
        event: None,
        comment: None,
        error: None,
    }
}

async fn run<'a>(
    parser: &str,
    chunks: impl IntoIterator<Item = &'a str>,
) -> Vec<Annotated<CreateChatCompletionStreamResponse>> {
    let chunks: Vec<_> = chunks.into_iter().map(chunk).collect();
    JailedStream::builder()
        .tool_call_parser(parser)
        .build()
        .apply_with_finish_reason(stream::iter(chunks))
        .collect()
        .await
}

fn tool_calls(
    responses: &[Annotated<CreateChatCompletionStreamResponse>],
) -> Vec<(String, String)> {
    responses
        .iter()
        .filter_map(|response| response.data.as_ref())
        .flat_map(|response| response.choices.iter())
        .filter_map(|choice| choice.delta.tool_calls.as_ref())
        .flatten()
        .filter_map(|call| call.function.as_ref())
        .filter_map(|function| {
            Some((
                function.name.clone()?,
                function.arguments.clone().unwrap_or_default(),
            ))
        })
        .collect()
}

fn content(responses: &[Annotated<CreateChatCompletionStreamResponse>]) -> String {
    responses
        .iter()
        .filter_map(|response| response.data.as_ref())
        .flat_map(|response| response.choices.iter())
        .filter_map(|choice| choice.delta.content.as_ref())
        .filter_map(|content| match content {
            ChatCompletionMessageContent::Text(text) => Some(text.as_str()),
            ChatCompletionMessageContent::Parts(_) => None,
        })
        .collect()
}

// Mistral v11 generation grammar: repeated [TOOL_CALLS]name[ARGS]JSON.
// Also accept the name{JSON} form used by existing deployments/vLLM.
#[tokio::test]
async fn mistral_v11_batch_and_jail_chunk_boundaries() {
    for (input, expected) in [
        (
            r#"[TOOL_CALLS]get_weather{"city":"Paris"}"#,
            vec![("get_weather", r#"{"city":"Paris"}"#)],
        ),
        (
            r#"[TOOL_CALLS]get_weather[ARGS]{"city":"東京 🧪","nested":{"values":[1,true,null]},"text":"} [TOOL_CALLS] [ARGS] [/TOOL_CALLS]"}"#,
            vec![(
                "get_weather",
                r#"{"city":"東京 🧪","nested":{"values":[1,true,null]},"text":"} [TOOL_CALLS] [ARGS] [/TOOL_CALLS]"}"#,
            )],
        ),
        (
            r#"[TOOL_CALLS]first[ARGS]{}[TOOL_CALLS]second{"x": 2}"#,
            vec![("first", "{}"), ("second", r#"{"x": 2}"#)],
        ),
    ] {
        let expected: Vec<_> = expected
            .into_iter()
            .map(|(n, a)| (n.to_owned(), a.to_owned()))
            .collect();
        let (calls, text) = try_tool_call_parse_aggregate_finalize(input, Some("mistral"), None)
            .await
            .unwrap();
        assert_eq!(
            calls
                .iter()
                .map(|c| (c.function.name.clone(), c.function.arguments.clone()))
                .collect::<Vec<_>>(),
            expected,
            "batch {input}"
        );
        assert_eq!(text.unwrap_or_default(), "");
        let mut chunkings = vec![vec![input]];
        // &str input forbids splitting a UTF-8 code point. Test every valid split.
        for (at, _) in input.char_indices().skip(1) {
            chunkings.push(vec![&input[..at], &input[at..]]);
        }
        chunkings.push(
            input
                .char_indices()
                .map(|(at, c)| &input[at..at + c.len_utf8()])
                .collect(),
        );
        for chunks in chunkings {
            let responses = run("mistral", chunks.clone()).await;
            assert_eq!(tool_calls(&responses), expected, "chunks {chunks:?}");
            assert_eq!(content(&responses), "", "chunks {chunks:?}");
            let wire_calls: Vec<_> = responses
                .iter()
                .filter_map(|r| r.data.as_ref())
                .flat_map(|r| &r.choices)
                .filter_map(|c| c.delta.tool_calls.as_ref())
                .flatten()
                .collect();
            assert_eq!(
                wire_calls.iter().map(|c| c.index).collect::<Vec<_>>(),
                (0..expected.len() as u32).collect::<Vec<_>>()
            );
            let ids: std::collections::HashSet<_> = wire_calls
                .iter()
                .map(|c| c.id.as_deref().unwrap())
                .collect();
            assert_eq!(ids.len(), expected.len());
        }
    }
}

#[tokio::test]
async fn mistral_v11_preserves_prose_and_drops_incomplete_calls() {
    for (input, expected_calls, expected_text) in [
        (
            "before [TOOL_CALLS] get_time[ARGS] {} after",
            vec![("get_time", "{}")],
            "before  after",
        ),
        (
            "[TOOL_CALLS]a{} between [TOOL_CALLS]b{} after",
            vec![("a", "{}"), ("b", "{}")],
            " between  after",
        ),
        (
            "[TOOL_CALLS]unknown_tool[ARGS]{}",
            vec![("unknown_tool", "{}")],
            "",
        ),
        ("[TOOL_CALLS]name", vec![], ""),
        ("[TOOL_CALLS]name[ARGS]", vec![], ""),
        (r#"[TOOL_CALLS]name{"x":"unfinished"#, vec![], ""),
        (r#"[TOOL_CALLS]name{"x":1"#, vec![], ""),
        (r#"[TOOL_CALLS]name{"x": invalid}"#, vec![], ""),
        ("[TOOL_CALLS][ARGS]{}", vec![], ""),
        ("[TOOL_CALLS]name[ARGS][]", vec![], ""),
        (
            r#"[TOOL_CALLS]done{}[TOOL_CALLS]cut{"x":"unfinished"#,
            vec![("done", "{}")],
            "",
        ),
    ] {
        let expected_calls: Vec<_> = expected_calls
            .into_iter()
            .map(|(n, a)| (n.to_owned(), a.to_owned()))
            .collect();
        let (calls, text) = try_tool_call_parse_aggregate_finalize(input, Some("mistral"), None)
            .await
            .unwrap();
        assert_eq!(
            calls
                .into_iter()
                .map(|c| (c.function.name, c.function.arguments))
                .collect::<Vec<_>>(),
            expected_calls,
            "batch {input}"
        );
        assert_eq!(text.unwrap_or_default(), expected_text, "batch {input}");
        let tiny = input
            .char_indices()
            .map(|(at, c)| &input[at..at + c.len_utf8()]);
        let responses = run("mistral", tiny).await;
        assert_eq!(
            tool_calls(&responses),
            expected_calls,
            "tiny chunks: {input}"
        );
        assert_eq!(content(&responses), expected_text, "tiny chunks: {input}");
        for at in input.char_indices().map(|(i, _)| i).chain([input.len()]) {
            let responses = run("mistral", [&input[..at], &input[at..]]).await;
            assert_eq!(
                tool_calls(&responses),
                expected_calls,
                "split {at}: {input}"
            );
            assert_eq!(content(&responses), expected_text, "split {at}: {input}");
        }
    }
}

#[tokio::test]
async fn mistral_v11_length_finish_keeps_only_complete_calls() {
    use dynamo_protocols::types::FinishReason;
    let mut terminal = chunk(r#"[TOOL_CALLS]cut[ARGS]{"city":"Par"#);
    terminal.data.as_mut().unwrap().choices[0].finish_reason = Some(FinishReason::Length);
    let responses: Vec<_> = JailedStream::builder()
        .tool_call_parser("mistral")
        .build()
        .apply_with_finish_reason(stream::iter([
            chunk(r#"[TOOL_CALLS]done[ARGS]{ "z":1e+02, "a":"escaped\\quote\"" }"#),
            terminal,
        ]))
        .collect()
        .await;
    assert_eq!(
        tool_calls(&responses),
        vec![(
            "done".into(),
            r#"{ "z":1e+02, "a":"escaped\\quote\"" }"#.into()
        )]
    );
    assert_eq!(content(&responses), "");
    assert_eq!(
        responses
            .iter()
            .filter_map(|r| r.data.as_ref())
            .flat_map(|r| &r.choices)
            .filter_map(|c| c.finish_reason)
            .next_back(),
        Some(FinishReason::Length)
    );
}

#[tokio::test]
async fn mistral_v11_unknown_name_is_not_schema_validation() {
    use dynamo_parsers::tool_calling::ToolDefinition;
    let tools = [ToolDefinition {
        name: "known".into(),
        parameters: None,
        strict: None,
    }];
    let (calls, _) = try_tool_call_parse_aggregate_finalize(
        "[TOOL_CALLS]unknown[ARGS]{}",
        Some("mistral"),
        Some(&tools),
    )
    .await
    .unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "unknown");
}
