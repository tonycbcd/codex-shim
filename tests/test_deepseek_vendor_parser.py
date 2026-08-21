from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_node_json(script: str) -> dict:
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_codeproxy_bridge_preserves_history_reasoning_namespace_and_usage():
    result = _run_node_json(
        r"""
import { streamCodeproxyResponses } from './vendor/deepseek-web-api/dist/api/codeproxyResponses.js';
let upstreamBody;
const client = {
  async streamChat(body, emit) {
    upstreamBody = body;
    emit({id:'chat_1',object:'chat.completion.chunk',created:1,model:'deepseek',choices:[{index:0,delta:{reasoning_content:'inspect'}}]});
    emit({id:'chat_1',object:'chat.completion.chunk',created:1,model:'deepseek',choices:[{index:0,delta:{tool_calls:[
      {index:0,id:'call_a',type:'function',function:{name:'codegraph_explore',arguments:'{"query":"verified"}'}},
      {index:1,id:'call_b',type:'function',function:{name:'exec_command',arguments:'{"cmd":"php -l file.php"}'}}
    ]}}]});
    emit({id:'chat_1',object:'chat.completion.chunk',created:1,model:'deepseek',choices:[{index:0,delta:{},finish_reason:'tool_calls'}],usage:{prompt_tokens:11,completion_tokens:7,total_tokens:18,prompt_tokens_details:{cached_tokens:3}}});
  }
};
const body = {
  model:'deepseek',
  input:[
    {role:'user',content:[{type:'input_text',text:'first'}]},
    {type:'function_call',call_id:'old_call',name:'exec_command',arguments:'{"cmd":"pwd"}'},
    {type:'function_call_output',call_id:'old_call',output:'"/repo"'},
    {role:'user',content:'continue'}
  ],
  tools:[
    {type:'function',name:'exec_command',parameters:{type:'object',properties:{cmd:{type:'string'}},required:['cmd']}},
    {type:'namespace',name:'mcp__codegraph__',tools:[
      {type:'function',name:'codegraph_explore',parameters:{type:'object',properties:{query:{type:'string'}},required:['query']}}
    ]}
  ]
};
const events=[]; for await (const event of streamCodeproxyResponses(body,client)) events.push(event);
const done=events.filter(event=>event.type==='response.output_item.done').map(event=>event.item);
const completed=events.find(event=>event.type==='response.completed');
console.log(JSON.stringify({
  upstreamBody,
  types:events.map(event=>event.type),
  done,
  usage:completed.response.usage
}));
"""
    )

    messages = result["upstreamBody"]["messages"]
    assert messages[0]["role"] == "user"
    assert any(message["role"] == "assistant" and message.get("tool_calls") for message in messages)
    assert any(message["role"] == "tool" and message["tool_call_id"] == "old_call" for message in messages)
    assert result["upstreamBody"]["tools"][1]["function"]["name"].startswith(
        "mcp__codegraph__"
    )
    assert "response.reasoning_text.delta" in result["types"]
    calls = [item for item in result["done"] if item["type"] == "function_call"]
    assert len(calls) == 2
    assert calls[0]["name"] == "codegraph_explore"
    assert calls[0]["namespace"] == "mcp__codegraph__"
    assert calls[1]["name"] == "exec_command"
    assert result["usage"]["input_tokens"] == 11
    assert result["usage"]["output_tokens"] == 7
    assert result["usage"]["input_tokens_details"]["cached_tokens"] == 3


def test_codeproxy_bridge_restores_custom_apply_patch_events():
    result = _run_node_json(
        r"""
import { streamCodeproxyResponses } from './vendor/deepseek-web-api/dist/api/codeproxyResponses.js';
let upstreamBody;
const client = {
  async streamChat(body, emit) {
    upstreamBody = body;
    emit({id:'chat_2',object:'chat.completion.chunk',created:1,model:'deepseek',choices:[{index:0,delta:{tool_calls:[{
      index:0,id:'patch_1',type:'function',function:{name:'apply_patch',arguments:'{"patch":"*** Begin Patch\\n*** End Patch"}'}
    }]}}]});
    emit({id:'chat_2',object:'chat.completion.chunk',created:1,model:'deepseek',choices:[{index:0,delta:{},finish_reason:'tool_calls'}],usage:{prompt_tokens:1,completion_tokens:1,total_tokens:2}});
  }
};
const body={model:'deepseek',input:'patch it',tools:[{type:'apply_patch'}]};
const events=[]; for await (const event of streamCodeproxyResponses(body,client)) events.push(event);
console.log(JSON.stringify({upstreamBody,events}));
"""
    )

    upstream_tool = result["upstreamBody"]["tools"][0]["function"]
    assert upstream_tool["name"] == "apply_patch"
    assert upstream_tool["parameters"]["required"] == ["patch"]
    event_types = [event["type"] for event in result["events"]]
    assert "response.custom_tool_call_input.delta" in event_types
    assert "response.custom_tool_call_input.done" in event_types
    done = next(
        event
        for event in result["events"]
        if event["type"] == "response.output_item.done"
    )
    assert done["item"]["type"] == "custom_tool_call"
    assert done["item"]["input"].startswith("*** Begin Patch")
    completed = next(
        event for event in result["events"] if event["type"] == "response.completed"
    )
    assert completed["response"]["output"][0]["type"] == "custom_tool_call"


def test_codeproxy_bridge_restores_deepseek_source_labels():
    result = _run_node_json(
        r"""
import { completeCodeproxyResponses, streamCodeproxyResponses } from './vendor/deepseek-web-api/dist/api/codeproxyResponses.js';
function clientFor(text) {
  return {
    async streamChat(body, emit) {
      emit({id:'chat_text',object:'chat.completion.chunk',created:1,model:body.model,choices:[{index:0,delta:{content:text}}]});
      emit({id:'chat_text',object:'chat.completion.chunk',created:1,model:body.model,choices:[{index:0,delta:{},finish_reason:'stop'}],usage:{prompt_tokens:1,completion_tokens:1,total_tokens:2}});
    }
  };
}
const streamEvents=[];
for await (const event of streamCodeproxyResponses(
  {model:'deepseek-v4-flash',stream:true,input:'Hello'},
  clientFor('你好')
)) streamEvents.push(event);
const proResponse=await completeCodeproxyResponses(
  {model:'deepseek-v4-pro',stream:false,input:'Hello'},
  clientFor('你好')
);
console.log(JSON.stringify({streamEvents,proResponse}));
"""
    )

    delta = next(
        event
        for event in result["streamEvents"]
        if event["type"] == "response.output_text.delta"
    )
    assert delta["delta"] == "[deepseek] 你好"
    stream_done = next(
        event
        for event in result["streamEvents"]
        if event["type"] == "response.output_item.done"
    )
    assert stream_done["item"]["content"][0]["text"] == "[deepseek] 你好"
    assert (
        result["proResponse"]["output"][0]["content"][0]["text"]
        == "[deepseek-pro] 你好"
    )


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


def test_vendor_parser_recovers_broken_invoke_open_tag():
    script = r"""
import { parseToolCalls } from './vendor/deepseek-web-api/dist/deepseek/toolCalls.js';
const definitions = [{
  type: 'function',
  name: 'exec_command',
  parameters: {
    type: 'object',
    properties: {
      cmd: { type: 'string' },
      workdir: { type: 'string' },
      yield_time_ms: { type: 'integer' },
      max_output_tokens: { type: 'integer' }
    },
    required: ['cmd']
  }
}];
const parsed = parseToolCalls(
  `<oke name="exec_command">
<parameter name="cmd">cd /opt/codes/ms/front && git diff -- src/common/i18n/mobile/en.js | cat</parameter>
<parameter name="workdir">/opt/codes/ms/front</parameter>
<parameter name="yield_time_ms">1000</parameter>
<parameter name="max_output_tokens">12000</parameter>
</invoke>`,
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
        "cmd": "cd /opt/codes/ms/front && git diff -- src/common/i18n/mobile/en.js | cat",
        "workdir": "/opt/codes/ms/front",
        "yield_time_ms": 1000,
        "max_output_tokens": 12000,
    }


def test_vendor_parser_recovers_exec_command_closed_as_exec_calls():
    script = r"""
import { parseToolCalls } from './vendor/deepseek-web-api/dist/deepseek/toolCalls.js';
const definitions = [{
  type: 'function',
  name: 'exec_command',
  parameters: {
    type: 'object',
    properties: {
      cmd: { type: 'string' },
      justification: { type: 'string' },
      login: { type: 'boolean' },
      max_output_tokens: { type: 'integer' },
      prefix_rule: { type: 'array', items: { type: 'string' } },
      sandbox_permissions: { type: 'string' },
      shell: { type: 'string' },
      tty: { type: 'boolean' },
      workdir: { type: 'string' },
      yield_time_ms: { type: 'integer' }
    },
    required: ['cmd']
  }
}];
const parsed = parseToolCalls(
  `<exec_command>
<cmd>grep -nEi "block|ban|blacklist|whitelist|allowlist|restrict|wanted|forbid|deny|disable|prohibit|not.?allow|isFromWantedCountry|CountryRestriction" /opt/codes/ms/de4-web/common/controllers/censor/ProfileController.php</cmd>
<justification>Inspect for any country-based blocking logic</justification>
<login>true</login>
<max_output_tokens>20000</max_output_tokens>
<prefix_rule>[]</prefix_rule>
<sandbox_permissions>use_default</sandbox_permissions>
<shell>bash</shell>
<tty>false</tty>
<workdir>/opt/codes/ms-yq2anvut27d66c7yhr7lu</workdir>
<yield_time_ms>10000</yield_time_ms>
</exec_calls>`,
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
        "cmd": 'grep -nEi "block|ban|blacklist|whitelist|allowlist|restrict|wanted|forbid|deny|disable|prohibit|not.?allow|isFromWantedCountry|CountryRestriction" /opt/codes/ms/de4-web/common/controllers/censor/ProfileController.php',
        "justification": "Inspect for any country-based blocking logic",
        "login": True,
        "max_output_tokens": 20000,
        "prefix_rule": [],
        "sandbox_permissions": "use_default",
        "shell": "bash",
        "tty": False,
        "workdir": "/opt/codes/ms-yq2anvut27d66c7yhr7lu",
        "yield_time_ms": 10000,
    }


def test_vendor_parser_recovers_mismatched_command_wrapper_tags():
    script = r"""
import { parseToolCalls } from './vendor/deepseek-web-api/dist/deepseek/toolCalls.js';
const definitions = [{
  type: 'function',
  name: 'exec_command',
  parameters: {
    type: 'object',
    properties: {
      cmd: { type: 'string' },
      justification: { type: 'string' }
    },
    required: ['cmd']
  }
}];
const text = `<_command>
<cmd>cat /opt/codes/ms/de4-web/common/services/Console/ScamAnalysisService.php</cmd>
<justification>Read the ScamAnalysisService class to understand its methods and usage</justification>
</exec_command>`;
console.log(JSON.stringify(parseToolCalls(text, 'test', definitions)));
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
        "cmd": (
            "cat /opt/codes/ms/de4-web/common/services/Console/"
            "ScamAnalysisService.php"
        ),
        "justification": (
            "Read the ScamAnalysisService class to understand its methods and usage"
        ),
    }


def test_vendor_parser_recovers_flat_json_arguments_inside_named_tool_tags():
    script = r"""
import { parseToolCalls } from './vendor/deepseek-web-api/dist/deepseek/toolCalls.js';
const definitions = [{
  type: 'function',
  name: 'exec_command',
  parameters: {
    type: 'object',
    properties: {
      cmd: { type: 'string' },
      max_output_tokens: { type: 'integer' }
    },
    required: ['cmd']
  }
}];
const text = `<_call name="exec_command">
{"cmd":"find /tmp -name '*.php'","max_output_tokens":2000}
</tool_call>
<tool_call name="exec_command">
{"cmd":"git status","max_output_tokens":2000}
</tool_call>`;
console.log(JSON.stringify(parseToolCalls(text, 'test', definitions)));
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
    assert [
        json.loads(call["function"]["arguments"])
        for call in parsed["toolCalls"]
    ] == [
        {
            "cmd": "find /tmp -name '*.php'",
            "max_output_tokens": 2000,
        },
        {
            "cmd": "git status",
            "max_output_tokens": 2000,
        },
    ]


def test_vendor_parser_recovers_codegraph_name_embedded_in_arguments():
    script = r"""
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
const text = `<_call>
{"arguments":{"input":"verified photo delete user status censor","name":"mcp__codegraph"}
</tool_call>`;
console.log(JSON.stringify(parseToolCalls(text, 'test', definitions)));
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
        "cmd": "codegraph explore 'verified photo delete user status censor'",
    }


def test_vendor_parser_recovers_code_fenced_json_tool_call_array():
    script = r"""
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
const text = `\`\`\`
[
{
"name": "mcp__codegraph",
"arguments": {
"input": "VERIFY_PHOTO_TIMELINE deletePhotos censor"
}
}
]
\`\`\``;
console.log(JSON.stringify(parseToolCalls(text, 'test', definitions)));
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
        "cmd": "codegraph explore 'VERIFY_PHOTO_TIMELINE deletePhotos censor'",
    }


def test_vendor_parser_recovers_json_tool_fence_after_explanatory_text():
    script = r"""
import { parseToolCalls } from './vendor/deepseek-web-api/dist/deepseek/toolCalls.js';
const definitions = [{
  type: 'function',
  name: 'exec_command',
  parameters: {
    type: 'object',
    properties: {
      cmd: { type: 'string' },
      justification: { type: 'string' },
      workdir: { type: 'string' }
    },
    required: ['cmd']
  }
}];
const text = `好的CodeGraph 不可用。我先通过常规工具定位相关代码。

\`\`\`json
[
  {
    "name": "exec_command",
    "arguments": {
      "cmd": "find /workspace -name '*.php'",
      "justification": "Locate censor logic",
      "workdir": "/workspace"
    }
  }
]
\`\`\``;
console.log(JSON.stringify(parseToolCalls(text, 'test', definitions)));
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    parsed = json.loads(completed.stdout)
    assert parsed["content"] == "好的CodeGraph 不可用。我先通过常规工具定位相关代码。"
    assert len(parsed["toolCalls"]) == 1
    call = parsed["toolCalls"][0]["function"]
    assert call["name"] == "exec_command"
    assert json.loads(call["arguments"]) == {
        "cmd": "find /workspace -name '*.php'",
        "justification": "Locate censor logic",
        "workdir": "/workspace",
    }


def test_vendor_parser_maps_codegraph_input_to_query_for_concrete_tool():
    script = r"""
import { parseToolCalls } from './vendor/deepseek-web-api/dist/deepseek/toolCalls.js';
const definitions = [{
  type: 'function',
  name: 'codegraph_explore',
  parameters: {
    type: 'object',
    properties: {
      projectPath: { type: 'string' },
      query: { type: 'string' },
      maxFiles: { type: 'integer' }
    },
    required: ['query'],
    additionalProperties: false
  }
}];
const text = `<tool_call>{"name":"mcp__codegraph","arguments":{"input":"deletePhotos verified status","projectPath":"/workspace/de4-web"}}</tool_call>`;
console.log(JSON.stringify(parseToolCalls(text, 'test', definitions)));
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    parsed = json.loads(completed.stdout)
    assert len(parsed["toolCalls"]) == 1
    call = parsed["toolCalls"][0]["function"]
    assert call["name"] == "codegraph_explore"
    assert json.loads(call["arguments"]) == {
        "projectPath": "/workspace/de4-web",
        "query": "deletePhotos verified status",
    }
