from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_vendor_parser_infers_exec_command_for_arguments_only_call():
    script = """
import { parseToolCalls } from './vendor/deepseek-web-api/dist/deepseek/toolCalls.js';
const definitions = [
  {
    type: 'function',
    name: 'exec_command',
    parameters: {
      type: 'object',
      properties: {
        cmd: { type: 'string' },
        workdir: { type: 'string' },
        max_output_tokens: { type: 'integer' }
      },
      required: ['cmd']
    }
  },
  {
    type: 'function',
    name: 'write_stdin',
    parameters: {
      type: 'object',
      properties: {
        session_id: { type: 'integer' },
        chars: { type: 'string' }
      },
      required: ['session_id']
    }
  }
];
const parsed = parseToolCalls(
  '<_call>\\n{"arguments":{"cmd":"pwd","max_output_tokens":20000}}\\n</tool_call>',
  'test',
  definitions
);
console.log(JSON.stringify(parsed));
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    parsed = json.loads(completed.stdout)
    assert parsed["content"] == ""
    assert len(parsed["toolCalls"]) == 1
    call = parsed["toolCalls"][0]["function"]
    assert call["name"] == "exec_command"
    assert json.loads(call["arguments"]) == {
        "cmd": "pwd",
        "max_output_tokens": 20000,
    }


def test_vendor_parser_infers_tool_for_bare_arguments_json():
    script = """
import { parseToolCalls } from './vendor/deepseek-web-api/dist/deepseek/toolCalls.js';
const definitions = [{
  type: 'function',
  name: 'exec_command',
  parameters: {
    type: 'object',
    properties: { cmd: { type: 'string' } },
    required: ['cmd']
  }
}];
console.log(JSON.stringify(parseToolCalls('{"arguments":{"cmd":"git status"}}', 'test', definitions)));
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    parsed = json.loads(completed.stdout)
    assert parsed["toolCalls"][0]["function"]["name"] == "exec_command"


def test_vendor_parser_recovers_tool_call_missing_opening_angle_bracket():
    script = """
import { parseToolCalls } from './vendor/deepseek-web-api/dist/deepseek/toolCalls.js';
const definitions = [{
  type: 'function',
  name: 'exec_command',
  parameters: {
    type: 'object',
    properties: {
      cmd: { type: 'string' },
      workdir: { type: 'string' }
    },
    required: ['cmd']
  }
}];
const parsed = parseToolCalls(
  'tool_call>\\n{"arguments":{"cmd":"pwd","workdir":"/tmp"},"name":"exec_command"}\\n</tool_call>',
  'test',
  definitions
);
console.log(JSON.stringify(parsed));
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    parsed = json.loads(completed.stdout)
    assert parsed["content"] == ""
    assert len(parsed["toolCalls"]) == 1
    call = parsed["toolCalls"][0]["function"]
    assert call["name"] == "exec_command"
    assert json.loads(call["arguments"]) == {
        "cmd": "pwd",
        "workdir": "/tmp",
    }
