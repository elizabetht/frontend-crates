// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Emit real builder output for scripts/validate_response_schema.py.
use dynamo_parsers_v2::structural_tag::{
    ReasoningBoundary, StructuralTagContext, StructuralTagOptions, StructuralTagSchemaMode,
    StructuralTagToolChoice,
};
use dynamo_parsers_v2::{Tool, structural_tag_builder_for_family};
use serde_json::json;

fn main() -> anyhow::Result<()> {
    let tools = [Tool {
        name: "lookup".into(),
        description: None,
        parameters: json!({"type": "object"}),
        strict: Some(true),
    }];
    let mut cases = Vec::new();
    for family in ["qwen3_coder", "deepseek_v4", "glm47"] {
        let builder = structural_tag_builder_for_family(family).unwrap();
        for (choice, tool_choice, tools) in [
            (
                "none_with_tools",
                StructuralTagToolChoice::None,
                tools.as_slice(),
            ),
            ("none_without_tools", StructuralTagToolChoice::None, &[]),
            ("auto_without_tools", StructuralTagToolChoice::Auto, &[]),
        ] {
            for (reasoning, starts_in_reasoning, reasoning_boundary) in [
                ("off", false, ReasoningBoundary::StructuralTag),
                ("owned", true, ReasoningBoundary::StructuralTag),
                ("external", true, ReasoningBoundary::External),
            ] {
                for (schema_name, schema, valid, invalid) in [
                    (
                        "object",
                        json!({
                            "type": "object",
                            "properties": {"answer": {"type": "integer"}},
                            "required": ["answer"],
                            "additionalProperties": false
                        }),
                        vec![r#"{"answer":42}"#, r#"{"answer":-1}"#],
                        vec![
                            r#"{"answer":"wrong"}"#,
                            "{}",
                            r#"{"answer":42,"extra":0}"#,
                            "null",
                            "plain text",
                        ],
                    ),
                    (
                        "scalar",
                        json!({"type":"string", "enum":["yes", "no"]}),
                        vec![r#""yes""#, r#""no""#],
                        vec![r#""maybe""#, "42", "{}", "plain text"],
                    ),
                ] {
                    let prefix = if reasoning == "owned" {
                        if family == "qwen3_coder" {
                            "Checking.</think>\n\n"
                        } else {
                            "Checking.</think>"
                        }
                    } else {
                        ""
                    };
                    let structural_tag = builder.build_with_options(
                        &StructuralTagContext {
                            tool_choice,
                            tools,
                            parallel_tool_calls: None,
                            schema_mode: StructuralTagSchemaMode::Auto,
                            structured_output_schema: Some(&schema),
                            starts_in_reasoning,
                        },
                        &StructuralTagOptions {
                            reasoning_boundary,
                            tool_arguments_any_order: true,
                            ..Default::default()
                        },
                    )?;
                    cases.push(json!({
                        "name": format!("{family}/{choice}/{reasoning}/{schema_name}"),
                        "structural_tag": structural_tag,
                        "accept": valid.iter().map(|s| format!("{prefix}{s}")).collect::<Vec<_>>(),
                        "reject": invalid.iter().map(|s| format!("{prefix}{s}")).collect::<Vec<_>>(),
                    }));
                }
            }
        }
    }
    println!("{}", serde_json::to_string(&cases)?);
    Ok(())
}
