// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Run with `cargo run -p dynamo-renderer --example qwen3_thinking`.
//! Uses the pinned Qwen3-0.6B template documented in fixtures/README.md.
//! This exercises prompt rendering, without loading a model or executing tools.

use std::collections::HashMap;

use dynamo_protocols::types::CreateChatCompletionRequest;
use dynamo_renderer::{ChatTemplate, ContextMixins, OAIChatLikeRequest, PromptFormatter};
use serde_json::{Value, json};

// Consumer-owned extensions reach the HF template through this trait method.
// The standard OpenAI request does not expose chat_template_args.
struct Request {
    chat: CreateChatCompletionRequest,
    template_args: HashMap<String, Value>,
}

impl OAIChatLikeRequest for Request {
    fn model(&self) -> String {
        self.chat.model()
    }

    fn messages(&self) -> minijinja::Value {
        self.chat.messages()
    }

    fn tools(&self) -> Option<minijinja::Value> {
        self.chat.tools()
    }

    fn tool_choice(&self) -> Option<minijinja::Value> {
        self.chat.tool_choice()
    }

    fn reasoning_effort(&self) -> Option<minijinja::Value> {
        self.chat.reasoning_effort()
    }

    fn should_add_generation_prompt(&self) -> bool {
        true
    }

    fn chat_template_args(&self) -> Option<&HashMap<String, Value>> {
        Some(&self.template_args)
    }
}

fn render(template_args: HashMap<String, Value>, effort: Option<&str>) -> anyhow::Result<String> {
    let mut chat = json!({
        "model": "Qwen/Qwen3-0.6B",
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        }]
    });
    if let Some(effort) = effort {
        chat["reasoning_effort"] = json!(effort);
    }
    let request = Request {
        chat: serde_json::from_value(chat)?,
        template_args,
    };
    let template: ChatTemplate = serde_json::from_value(json!({
        "chat_template": include_str!("fixtures/qwen3-0.6b.jinja")
    }))?;
    let PromptFormatter::OAI(formatter) =
        PromptFormatter::from_parts(template, ContextMixins::default(), false)?;
    formatter.render(&request)
}

fn thinking_args(enabled: bool) -> HashMap<String, Value> {
    HashMap::from([("enable_thinking".to_owned(), json!(enabled))])
}

fn main() -> anyhow::Result<()> {
    for enabled in [true, false] {
        let prompt = render(thinking_args(enabled), None)?;
        println!("enable_thinking={enabled}\n{prompt}\n");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabling_thinking_prefills_an_empty_reasoning_block() {
        let enabled = render(thinking_args(true), None).unwrap();
        let disabled = render(thinking_args(false), None).unwrap();
        assert!(enabled.ends_with("<|im_start|>assistant\n"));
        assert_eq!(disabled, format!("{enabled}<think>\n\n</think>\n\n"));
        assert_eq!(render(HashMap::new(), None).unwrap(), enabled);
        // The tool instruction remains present in both modes. Disabling
        // thinking does not disable the JSON-envelope tool-call protocol.
        assert!(enabled.contains("<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"));
        assert!(enabled.contains("get_weather"));
    }

    #[test]
    fn hf_template_does_not_translate_reasoning_effort_or_thinking_alias() {
        let enabled = render(thinking_args(true), None).unwrap();
        assert_eq!(render(HashMap::new(), Some("none")).unwrap(), enabled);
        assert_eq!(
            render(HashMap::from([("thinking".to_owned(), json!(false))]), None).unwrap(),
            enabled
        );
        assert_eq!(
            render(thinking_args(false), Some("high")).unwrap(),
            render(thinking_args(false), None).unwrap()
        );
    }
}
