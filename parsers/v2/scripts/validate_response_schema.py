#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate this checkout's structural tags using XGrammar in a cluster pod.

Usage: python3 parsers/v2/scripts/validate_response_schema.py --pod POD [--namespace NS]
Only starts a CPU grammar check in the existing container. Does not alter its server.
"""
import argparse
import json
from pathlib import Path
import subprocess

CHECK = r'''
import importlib.metadata
import json
import sys
import socket
import xgrammar as xgr

cases = json.load(sys.stdin)
# Byte vocabulary makes every tested character a token, including punctuation.
info = xgr.TokenizerInfo([bytes([i]) for i in range(256)] + [b"<eos>"], stop_token_ids=[256])
compiler = xgr.GrammarCompiler(info, max_threads=1)
checks = 0
for case in cases:
    tag = case["structural_tag"]
    if tag is None:
        raise AssertionError(f'{case["name"]}: response schema was dropped')
    grammar = compiler.compile_structural_tag(json.dumps(tag))
    for expected, key in [(True, "accept"), (False, "reject")]:
        for text in case[key]:
            matcher = xgr.GrammarMatcher(grammar)
            accepted = all(matcher.accept_token(token) for token in text.encode("utf-8"))
            accepted = accepted and matcher.accept_token(256)
            if accepted != expected:
                raise AssertionError(f'{case["name"]}: {text!r}: accepted={accepted}, expected={expected}')
            checks += 1
print(json.dumps({"pod": socket.gethostname(), "xgrammar": importlib.metadata.version("xgrammar"), "cases": len(cases), "checks": checks, "result": "PASS"}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod", required=True)
    parser.add_argument("--namespace", default="token-labs")
    parser.add_argument("--container")
    parser.add_argument("--context", help="kubectl context; defaults to the current context")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["cargo", "run", "--locked", "--quiet", "-p", "dynamo-parsers-v2", "--example", "response_schema_cases"],
        cwd=root, check=True, stdout=subprocess.PIPE, text=True,
    )
    cases = json.loads(result.stdout)
    command = ["kubectl"]
    if args.context:
        command += ["--context", args.context]
    command += ["--namespace", args.namespace, "exec", "-i", args.pod]
    if args.container:
        command += ["--container", args.container]
    command += ["--", "python3", "-c", CHECK]
    result = subprocess.run(command, input=json.dumps(cases), text=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
