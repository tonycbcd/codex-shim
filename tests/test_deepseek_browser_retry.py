import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_node(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_prepare_completion_retries_destroyed_navigation_context():
    result = run_node(
        """
        import { prepareCompletion } from "./vendor/deepseek-web-api/dist/deepseek/pow.js";
        let calls = 0;
        let waits = 0;
        const page = {
          async evaluate() {
            calls += 1;
            if (calls === 1) {
              throw new Error("Execution context was destroyed, most likely because of a navigation.");
            }
            return { sessionId: "session-ok", token: "token", powHeader: "pow" };
          },
          async waitForLoadState() { waits += 1; },
          async waitForTimeout() {},
        };
        const value = await prepareCompletion({
          page,
          powWorkerUrl: "",
          modelType: "expert",
          fallbackToken: "token",
          sessionId: null,
          reuseSession: false,
        });
        console.log(JSON.stringify({ calls, waits, sessionId: value.sessionId }));
        """
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '{"calls":2,"waits":1,"sessionId":"session-ok"}'


def test_deepseek_client_serializes_shared_page_preparation():
    result = run_node(
        """
        import { DeepSeekClient } from "./vendor/deepseek-web-api/dist/deepseek/client.js";
        let active = 0;
        let maxActive = 0;
        const page = {
          async evaluate() {
            active += 1;
            maxActive = Math.max(maxActive, active);
            await new Promise((resolve) => setTimeout(resolve, 20));
            active -= 1;
            return { sessionId: crypto.randomUUID(), token: "token", powHeader: "pow" };
          },
        };
        const auth = { token: "token", cookie: "", cookies: [], dumped_at: "" };
        const login = {
          async dumpCurrent() { return auth; },
          async page() { return page; },
        };
        const sessions = {
          resolve() { return null; },
          has() { return false; },
          get() { return undefined; },
        };
        const logger = { debug() {}, info() {}, warn() {} };
        globalThis.fetch = async () => ({ body: null });
        const client = new DeepSeekClient(
          {
            baseUrl: "https://chat.deepseek.com",
            powWorkerUrl: "",
            toolReasoning: false,
          },
          login,
          sessions,
          logger,
        );
        const body = { model: "deepseek-v4-pro", messages: [{ role: "user", content: "test" }] };
        await Promise.all([client.prepare(body), client.prepare(body)]);
        console.log(JSON.stringify({ maxActive }));
        """
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '{"maxActive":1}'


def test_persistent_empty_upstream_resets_browser_then_raises_specific_error():
    result = run_node(
        """
        import { DeepSeekClient } from "./vendor/deepseek-web-api/dist/deepseek/client.js";
        let resetCalls = 0;
        let executeCalls = 0;
        const prompts = [];
        const page = {
          async evaluate() {
            return { sessionId: crypto.randomUUID(), token: "token", powHeader: "pow" };
          },
        };
        const auth = { token: "token", cookie: "", cookies: [], dumped_at: "" };
        const login = {
          async dumpCurrent() { return auth; },
          async page() { return page; },
          async resetBrowserConnection() { resetCalls += 1; return auth; },
        };
        const sessions = {
          resolve() { return null; },
          has() { return false; },
          get() { return undefined; },
        };
        const logger = { debug() {}, info() {}, warn() {} };
        globalThis.fetch = async (_url, options) => {
          prompts.push(JSON.parse(options.body).prompt);
          return { body: null };
        };
        const client = new DeepSeekClient(
          {
            baseUrl: "https://chat.deepseek.com",
            powWorkerUrl: "",
            toolReasoning: false,
          },
          login,
          sessions,
          logger,
        );
        const body = {
          model: "deepseek-v4-pro",
          messages: [
            { role: "user", content: "修复当前问题" },
            { role: "assistant", content: "X".repeat(20000) },
          ],
          tools: [{ type: "function", function: { name: "exec_command", parameters: {} } }],
        };
        const initialRun = {
          requestTurns: [
            { role: "user", content: "修复当前问题" },
            { role: "assistant", content: "X".repeat(20000) },
          ],
          latestUserText: "修复当前问题",
          instructionFingerprint: "",
          toolsFingerprint: "",
          toolDefinitions: body.tools,
          hasTools: true,
          retry: 0,
          sessionId: "initial",
          prompt: "initial",
        };
        const diagnostics = {
          recoverableEmpty: true,
          emptyUpstream: true,
          reasoningChars: 0,
          outputChars: 0,
          toolCallCount: 0,
        };
        let error = "";
        try {
          await client.withToolRecovery(body, initialRun, async () => {
            executeCalls += 1;
            return { diagnostics };
          });
        } catch (caught) {
          error = caught.message;
        }
        console.log(JSON.stringify({
          resetCalls,
          executeCalls,
          error,
          promptLengths: prompts.map((prompt) => prompt.length),
          strictHasLargeHistory: prompts[1]?.includes("XXXXX") ?? true,
        }));
        """
    )

    assert result.returncode == 0, result.stderr
    parsed = __import__("json").loads(result.stdout)
    assert parsed["resetCalls"] == 1
    assert parsed["executeCalls"] == 3
    assert parsed["error"] == (
        "DeepSeek Web returned an empty response after browser recovery "
        "and two automatic retries"
    )
    assert len(parsed["promptLengths"]) == 2
    assert parsed["promptLengths"][1] < parsed["promptLengths"][0]
    assert parsed["strictHasLargeHistory"] is False


def test_tool_turn_retries_short_action_preamble_without_tool_call():
    result = run_node(
        """
        import { resolveToolTurn } from "./vendor/deepseek-web-api/dist/deepseek/toolOutcome.js";
        const tools = [{ type: "function", function: { name: "exec_command" } }];
        const deferred = resolveToolTurn(
          "我先看一下当前代码状态和之前的改动内容。",
          "",
          "test",
          "",
          tools,
        );
        const finalAnswer = resolveToolTurn(
          "问题位于 SearchService.php，第 42 行需要将 double 显式转换为 long。",
          "",
          "test",
          "",
          tools,
        );
        console.log(JSON.stringify({
          deferredRecoverable: deferred.recoverableEmpty,
          deferredContent: deferred.parsed.content,
          finalRecoverable: finalAnswer.recoverableEmpty,
        }));
        """
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        '{"deferredRecoverable":true,"deferredContent":"","finalRecoverable":false}'
    )


def test_tool_turn_retries_chinese_deferred_action_after_planning_sentence():
    result = run_node(
        """
        import { resolveToolTurn } from "./vendor/deepseek-web-api/dist/deepseek/toolOutcome.js";
        const tools = [{ type: "function", function: { name: "exec_command" } }];
        const deferred = resolveToolTurn(
          '我们需要"Analyze Private Messages"按钮及其相关逻辑。首先搜索相关关键词。',
          "",
          "test",
          "",
          tools,
        );
        const shortDeferred = resolveToolTurn(
          "首先搜索相关关键词。",
          "",
          "test",
          "",
          tools,
        );
        const completedAnalysis = resolveToolTurn(
          "我搜索了相关代码，根因是 Risk Score 没有加入 warning 列表。",
          "",
          "test",
          "",
          tools,
        );
        console.log(JSON.stringify({
          deferredRecoverable: deferred.recoverableEmpty,
          deferredContent: deferred.parsed.content,
          shortDeferredRecoverable: shortDeferred.recoverableEmpty,
          completedAnalysisRecoverable: completedAnalysis.recoverableEmpty,
        }));
        """
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        '{"deferredRecoverable":true,"deferredContent":"",'
        '"shortDeferredRecoverable":true,"completedAnalysisRecoverable":false}'
    )


def test_tool_turn_retries_long_chinese_command_plan_without_tool_call():
    result = run_node(
        r"""
        import { resolveToolTurn } from "./vendor/deepseek-web-api/dist/deepseek/toolOutcome.js";
        const tools = [{ type: "function", function: { name: "exec_command" } }];
        const deferred = resolveToolTurn(
          `现在实现需求。我先查看当前的 ProfileController.php 中 actionIndex 方法和视图文件的现有结构，确保修改正确无误。

第一步：查看 actionIndex 方法
grep -n "function actionIndex" common/controllers/censor/ProfileController.php
（如果命令不可用，我会用其他方式定位，但这里先执行）

第二步：读取视图文件 profile.php 中的 warning 区域
grep -n -A 10 -B 2 "warning_div" common/views/censor/default/profile.php
请稍等，我马上执行检查。`,
          "",
          "test",
          "",
          tools,
        );
        const completedAnswer = resolveToolTurn(
          "修复已完成。必要时可执行 grep -n risk_score ProfileController.php 复核。",
          "",
          "test",
          "",
          tools,
        );
        console.log(JSON.stringify({
          deferredRecoverable: deferred.recoverableEmpty,
          deferredContent: deferred.parsed.content,
          completedRecoverable: completedAnswer.recoverableEmpty,
        }));
        """
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        '{"deferredRecoverable":true,"deferredContent":"",'
        '"completedRecoverable":false}'
    )


def test_tool_turn_retries_named_tool_intent_without_structured_call():
    result = run_node(
        """
        import { resolveToolTurn } from "./vendor/deepseek-web-api/dist/deepseek/toolOutcome.js";
        const tools = [{
          type: "function",
          function: { name: "codegraph_explore" },
        }];
        const outcome = resolveToolTurn(
          "我将 codegraph_explore 工具来探索 waitlist 相关的代码结构，以便理解其搜索业务逻辑。",
          "",
          "test",
          "",
          tools,
        );
        console.log(JSON.stringify({
          recoverable: outcome.recoverableEmpty,
          content: outcome.parsed.content,
          calls: outcome.parsed.toolCalls.length,
        }));
        """
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        '{"recoverable":true,"content":"","calls":0}'
    )


def test_tool_turn_keeps_action_text_when_it_contains_a_real_tool_call():
    result = run_node(
        """
        import { resolveToolTurn } from "./vendor/deepseek-web-api/dist/deepseek/toolOutcome.js";
        const tools = [{
          type: "function",
          function: {
            name: "exec_command",
            parameters: {
              type: "object",
              properties: { cmd: { type: "string" } },
              required: ["cmd"],
            },
          },
        }];
        const outcome = resolveToolTurn(
          '我先检查代码。\\n<tool_call>{"name":"exec_command","arguments":{"cmd":"git status --short"}}</tool_call>',
          "",
          "test",
          "",
          tools,
        );
        console.log(JSON.stringify({
          recoverable: outcome.recoverableEmpty,
          calls: outcome.parsed.toolCalls.length,
        }));
        """
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '{"recoverable":false,"calls":1}'


def test_tool_turn_recovers_nested_json_keyed_by_configured_tool_name():
    result = run_node(
        """
        import { resolveToolTurn } from "./vendor/deepseek-web-api/dist/deepseek/toolOutcome.js";
        const tools = [{
          type: "function",
          function: {
            name: "exec_command",
            parameters: {
              type: "object",
              properties: {
                cmd: { type: "string" },
                workdir: { type: "string" },
                yield_time_ms: { type: "integer" },
                max_output_tokens: { type: "integer" },
              },
              required: ["cmd"],
            },
          },
        }];
        const outcome = resolveToolTurn(
          `<_call>
{"exec_command":{"cmd":"pwd && ls -la","workdir":"/opt/codes/ms-yq2anvut27d66c7yhr7lu","yield_time_ms":10000,"max_output_tokens":4000}}
</tool_calls>`,
          "",
          "test",
          "",
          tools,
        );
        const call = outcome.parsed.toolCalls[0];
        console.log(JSON.stringify({
          recoverable: outcome.recoverableEmpty,
          calls: outcome.parsed.toolCalls.length,
          name: call?.function.name,
          arguments: call ? JSON.parse(call.function.arguments) : null,
        }));
        """
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        '{"recoverable":false,"calls":1,"name":"exec_command",'
        '"arguments":{"cmd":"pwd && ls -la","max_output_tokens":4000,'
        '"workdir":"/opt/codes/ms-yq2anvut27d66c7yhr7lu","yield_time_ms":10000}}'
    )


def test_tool_turn_recovers_multiple_xml_tools_with_typed_child_arguments():
    result = run_node(
        """
        import { resolveToolTurn } from "./vendor/deepseek-web-api/dist/deepseek/toolOutcome.js";
        const tools = [{
          type: "function",
          function: {
            name: "exec_command",
            parameters: {
              type: "object",
              properties: {
                cmd: { type: "string" },
                login: { type: "boolean" },
                max_output_tokens: { type: "integer" },
                prefix_rule: { type: "array", items: { type: "string" } },
                shell: { type: "string" },
                tty: { type: "boolean" },
                workdir: { type: "string" },
                yield_time_ms: { type: "integer" },
              },
              required: ["cmd"],
            },
          },
        }];
        const outcome = resolveToolTurn(
          `<tool_call>
<exec_command>
<cmd>git log --oneline -20</cmd>
<login>false</login>
<max_output_tokens>2000</max_output_tokens>
<prefix_rule>[]</prefix_rule>
<shell>bash</shell>
<tty>false</tty>
<workdir>/opt/codes/ms-yq2anvut27d66c7yhr7lu</workdir>
<yield_time_ms>10000</yield_time_ms>
</exec_command>
<exec_command>
<cmd>git status</cmd>
<login>false</login>
<max_output_tokens>2000</max_output_tokens>
<prefix_rule>[]</prefix_rule>
<shell>bash</shell>
<tty>false</tty>
<workdir>/opt/codes/ms-yq2anvut27d66c7yhr7lu</workdir>
<yield_time_ms>10000</yield_time_ms>
</exec_command>
</tool_calls>`,
          "",
          "test",
          "",
          tools,
        );
        console.log(JSON.stringify({
          recoverable: outcome.recoverableEmpty,
          calls: outcome.parsed.toolCalls.map((call) => ({
            name: call.function.name,
            arguments: JSON.parse(call.function.arguments),
          })),
        }));
        """
    )

    assert result.returncode == 0, result.stderr
    parsed = __import__("json").loads(result.stdout)
    assert parsed["recoverable"] is False
    assert len(parsed["calls"]) == 2
    assert [call["arguments"]["cmd"] for call in parsed["calls"]] == [
        "git log --oneline -20",
        "git status",
    ]
    first = parsed["calls"][0]
    assert first["name"] == "exec_command"
    assert first["arguments"]["login"] is False
    assert first["arguments"]["tty"] is False
    assert first["arguments"]["prefix_rule"] == []
    assert first["arguments"]["yield_time_ms"] == 10000


def test_tool_turn_recovers_native_deepseek_dsml_tool_call():
    result = run_node(
        """
        import { resolveToolTurn } from "./vendor/deepseek-web-api/dist/deepseek/toolOutcome.js";
        const tools = [{
          type: "function",
          function: {
            name: "exec_command",
            parameters: {
              type: "object",
              properties: {
                cmd: { type: "string" },
                max_output_tokens: { type: "integer" },
                workdir: { type: "string" },
                yield_time_ms: { type: "integer" },
              },
              required: ["cmd"],
            },
          },
        }];
        const outcome = resolveToolTurn(
          `<tool_calls>
<｜｜DSML｜｜invoke name="exec_command">
<｜｜DSML｜｜parameter name="cmd" string="true">git status</｜｜DSML｜｜parameter>
<｜｜DSML｜｜parameter name="max_output_tokens" string="false">4000</｜｜DSML｜｜parameter>
<｜｜DSML｜｜parameter name="workdir" string="true">/opt/codes/ms/search-service</｜｜DSML｜｜parameter>
<｜｜DSML｜｜parameter name="yield_time_ms" string="true">10000</｜｜DSML｜｜parameter>
</｜｜DSML｜｜invoke>
</｜｜DSML｜｜tool_calls>`,
          "",
          "test",
          "",
          tools,
        );
        console.log(JSON.stringify({
          recoverable: outcome.recoverableEmpty,
          content: outcome.parsed.content,
          calls: outcome.parsed.toolCalls.map((call) => ({
            name: call.function.name,
            arguments: JSON.parse(call.function.arguments),
          })),
        }));
        """
    )

    assert result.returncode == 0, result.stderr
    parsed = __import__("json").loads(result.stdout)
    assert parsed["recoverable"] is False
    assert parsed["content"] == ""
    assert parsed["calls"] == [
        {
            "name": "exec_command",
            "arguments": {
                "cmd": "git status",
                "max_output_tokens": 4000,
                "workdir": "/opt/codes/ms/search-service",
                "yield_time_ms": "10000",
            },
        }
    ]


def test_responses_tool_call_ids_are_unique_across_same_session_turns():
    result = run_node(
        """
        import { consumeResponses } from "./vendor/deepseek-web-api/dist/deepseek/mapResponses.js";
        const encoder = new TextEncoder();
        function upstream() {
          const body = new ReadableStream({
            start(controller) {
              controller.enqueue(encoder.encode(
                'event: ready\\ndata: {"request_message_id":1,"response_message_id":2}\\n\\n'
              ));
              controller.enqueue(encoder.encode(
                'data: {"v":{"response":{"fragments":[{"type":"RESPONSE","content":"<tool_call>\\\\n{\\\\\\"name\\\\\\":\\\\\\"exec_command\\\\\\",\\\\\\"arguments\\\\\\":{\\\\\\"cmd\\\\\\":\\\\\\"git status\\\\\\"}}\\\\n</tool_call>"}]}}}\\n\\n'
              ));
              controller.enqueue(encoder.encode('event: close\\ndata: {}\\n\\n'));
              controller.close();
            },
          });
          return new Response(body, { status: 200 });
        }
        const input = {
          sessionId: "11111111-1111-1111-1111-111111111111",
          publicModel: "deepseek",
          upstream: upstream(),
          toolCompatibilityEnabled: true,
          toolDefinitions: [{
            type: "function",
            function: { name: "exec_command", parameters: { type: "object" } },
          }],
          thinkingEnabled: true,
          searchEnabled: false,
          modelType: "default",
        };
        const first = await consumeResponses(input);
        const second = await consumeResponses({ ...input, upstream: upstream() });
        console.log(JSON.stringify({
          responseIdsDiffer: first.response.id !== second.response.id,
          callIdsDiffer: first.response.output[0].call_id !== second.response.output[0].call_id,
          firstResponseId: first.response.id,
        }));
        """
    )

    assert result.returncode == 0, result.stderr
    parsed = __import__("json").loads(result.stdout)
    assert parsed["responseIdsDiffer"] is True
    assert parsed["callIdsDiffer"] is True
    assert parsed["firstResponseId"].startswith(
        "resp_11111111-1111-1111-1111-111111111111_"
    )


def test_reused_session_compacts_unchanged_instructions_and_tools():
    result = run_node(
        """
        import { buildDeepSeekPrompt } from "./vendor/deepseek-web-api/dist/deepseek/promptBuild.js";
        const body = {
          instructions: "IMPORTANT ".repeat(2000),
          input: [{ role: "user", content: "fix it" }],
          tools: [{
            type: "function",
            function: {
              name: "exec_command",
              description: "RUN ".repeat(2000),
              parameters: {
                type: "object",
                properties: { cmd: { type: "string" } },
                required: ["cmd"],
              },
            },
          }],
        };
        const first = buildDeepSeekPrompt(body, { reusedSession: false });
        const reused = buildDeepSeekPrompt(body, {
          reusedSession: true,
          previous: {
            turns: [{ role: "assistant", content: "previous" }],
            instructionFingerprint: first.instructionFingerprint,
            toolsFingerprint: first.toolsFingerprint,
          },
        });
        console.log(JSON.stringify({
          firstLength: first.prompt.length,
          reusedLength: reused.prompt.length,
          hasCompactDefinitions: reused.prompt.includes("Compact definitions:"),
          replayedHugeDescription: reused.prompt.includes("RUN RUN RUN RUN RUN"),
        }));
        """
    )

    assert result.returncode == 0, result.stderr
    parsed = __import__("json").loads(result.stdout)
    assert parsed["reusedLength"] < parsed["firstLength"] / 5
    assert parsed["hasCompactDefinitions"] is True
    assert parsed["replayedHugeDescription"] is False


def test_tool_turn_parses_function_style_call_leaked_into_reasoning():
    result = run_node(
        r"""
        import { resolveToolTurn } from "./vendor/deepseek-web-api/dist/deepseek/toolOutcome.js";
        const reasoning = `我们需要继续查看完整方法。
<tool_call>
exec_command("sed -n '7397,8200p' /opt/codes/ms/de4php/service/connection-service/Modules/ConnectionModule.php", max_output_tokens=20000)
</tool_call>`;
        const outcome = resolveToolTurn(
          "",
          reasoning,
          "reasoning-function-style",
          "",
          [{
            type: "function",
            function: {
              name: "exec_command",
              parameters: {
                type: "object",
                properties: {
                  cmd: { type: "string" },
                  max_output_tokens: { type: "integer" },
                },
                required: ["cmd"],
              },
            },
          }],
        );
        console.log(JSON.stringify({
          recoverable: outcome.recoverableEmpty,
          content: outcome.parsed.content,
          calls: outcome.parsed.toolCalls.map((call) => ({
            name: call.function.name,
            arguments: JSON.parse(call.function.arguments),
          })),
        }));
        """
    )

    assert result.returncode == 0, result.stderr
    parsed = __import__("json").loads(result.stdout)
    assert parsed == {
        "recoverable": False,
        "content": "",
        "calls": [
            {
                "name": "exec_command",
                "arguments": {
                    "cmd": (
                        "sed -n '7397,8200p' "
                        "/opt/codes/ms/de4php/service/connection-service/"
                        "Modules/ConnectionModule.php"
                    ),
                    "max_output_tokens": 20000,
                },
            }
        ],
    }


def test_malformed_reasoning_tool_protocol_is_not_promoted_to_final_answer():
    result = run_node(
        r"""
        import { resolveToolTurn } from "./vendor/deepseek-web-api/dist/deepseek/toolOutcome.js";
        const outcome = resolveToolTurn(
          "",
          "我接下来修改代码。 <tool_call> exec_command(unclosed",
          "malformed-reasoning",
          "",
          [{ type: "function", function: { name: "exec_command", parameters: {} } }],
        );
        console.log(JSON.stringify({
          recoverable: outcome.recoverableEmpty,
          content: outcome.parsed.content,
          calls: outcome.parsed.toolCalls.length,
          promoted: outcome.promotedReasoning,
        }));
        """
    )

    assert result.returncode == 0, result.stderr
    assert __import__("json").loads(result.stdout) == {
        "recoverable": True,
        "content": "",
        "calls": 0,
        "promoted": False,
    }


def test_tool_call_missing_required_arguments_is_retried_not_executed():
    result = run_node(
        r"""
        import { resolveToolTurn } from "./vendor/deepseek-web-api/dist/deepseek/toolOutcome.js";
        const outcome = resolveToolTurn(
          '<tool_call>{"name":"exec_command","arguments":{}}</tool_call>',
          "",
          "missing-required",
          "",
          [{
            type: "function",
            name: "exec_command",
            parameters: {
              type: "object",
              properties: { cmd: { type: "string" } },
              required: ["cmd"],
            },
          }],
        );
        console.log(JSON.stringify({
          recoverable: outcome.recoverableEmpty,
          content: outcome.parsed.content,
          calls: outcome.parsed.toolCalls.length,
        }));
        """
    )

    assert result.returncode == 0, result.stderr
    assert __import__("json").loads(result.stdout) == {
        "recoverable": True,
        "content": "",
        "calls": 0,
    }


def test_completion_claiming_unapplied_diff_is_forced_to_retry():
    result = run_node(
        r"""
        import { DeepSeekClient } from "./vendor/deepseek-web-api/dist/deepseek/client.js";
        const warnings = [];
        const client = new DeepSeekClient(
          {},
          {},
          { get() { return { turns: [] }; } },
          { warn(message, fields) { warnings.push({ message, fields }); } },
        );
        const attempt = client.assessCompletion(
          {
            sessionId: "session-1",
            retry: 0,
            latestUserText: "修复 hidden profile 推荐问题",
            requestTurns: [],
            toolDefinitions: [{
              type: "function",
              name: "exec_command",
              parameters: {
                type: "object",
                properties: { cmd: { type: "string" } },
                required: ["cmd"],
              },
            }],
          },
          {
            diagnostics: {
              recoverableEmpty: false,
              toolCallCount: 0,
              emptyUpstream: false,
            },
            responseText: `已完成修改。

修改文件：ConnectionModule.php

补丁内容：
--- a/ConnectionModule.php
+++ b/ConnectionModule.php
@@ -1,2 +1,3 @@
+and u.hidden = 0`,
          },
        );
        console.log(JSON.stringify({
          recoverable: attempt.diagnostics.recoverableEmpty,
          incompleteMutation: attempt.diagnostics.incompleteMutation,
          warnings: warnings.length,
        }));
        """
    )

    assert result.returncode == 0, result.stderr
    assert __import__("json").loads(result.stdout) == {
        "recoverable": True,
        "incompleteMutation": True,
        "warnings": 1,
    }


def test_applied_patch_summary_is_not_forced_to_retry():
    result = run_node(
        r"""
        import { DeepSeekClient } from "./vendor/deepseek-web-api/dist/deepseek/client.js";
        const definitions = [{
          type: "function",
          name: "apply_patch",
          parameters: {
            type: "object",
            properties: { input: { type: "string" } },
            required: ["input"],
          },
        }, {
          type: "function",
          name: "exec_command",
          parameters: {
            type: "object",
            properties: { cmd: { type: "string" } },
            required: ["cmd"],
          },
        }];
        const client = new DeepSeekClient(
          {},
          {},
          {
            get() {
              return {
                turns: [{
                  role: "user",
                  content: "修复 hidden profile 推荐问题",
                }, {
                  role: "assistant",
                  content: '<tool_call>{"name":"apply_patch","arguments":{"input":"*** Begin Patch\\n*** End Patch"}}</tool_call>',
                }, {
                  role: "tool",
                  content: "Done!",
                }, {
                  role: "assistant",
                  content: '<tool_call>{"name":"exec_command","arguments":{"cmd":"git diff -- ConnectionModule.php"}}</tool_call>',
                }, {
                  role: "tool",
                  content: "Process exited with code 0\n+and u.hidden = 0",
                }],
              };
            },
          },
          { warn() { throw new Error("unexpected warning"); } },
        );
        const attempt = client.assessCompletion(
          {
            sessionId: "session-2",
            retry: 0,
            latestUserText: "修复 hidden profile 推荐问题",
            requestTurns: [],
            toolDefinitions: definitions,
          },
          {
            diagnostics: {
              recoverableEmpty: false,
              toolCallCount: 0,
              emptyUpstream: false,
            },
            responseText: `已完成修改。
补丁内容：
--- a/ConnectionModule.php
+++ b/ConnectionModule.php
@@ -1,2 +1,3 @@
+and u.hidden = 0`,
          },
        );
        console.log(JSON.stringify({
          recoverable: attempt.diagnostics.recoverableEmpty,
          incompleteMutation: attempt.diagnostics.incompleteMutation ?? false,
        }));
        """
    )

    assert result.returncode == 0, result.stderr
    assert __import__("json").loads(result.stdout) == {
        "recoverable": False,
        "incompleteMutation": False,
    }


def test_plain_completed_change_summary_without_diff_is_forced_to_retry():
    result = run_node(
        r"""
        import { DeepSeekClient } from "./vendor/deepseek-web-api/dist/deepseek/client.js";
        const client = new DeepSeekClient(
          {},
          {},
          { get() { return { turns: [] }; } },
          { warn() {} },
        );
        const attempt = client.assessCompletion(
          {
            sessionId: "session-summary",
            retry: 0,
            latestUserText: "修复删除最后一张公开照片后 verified 未取消的问题",
            requestTurns: [],
            toolDefinitions: [],
          },
          {
            diagnostics: {
              recoverableEmpty: false,
              toolCallCount: 0,
              emptyUpstream: false,
            },
            responseText: `已修复该问题。
修改文件：PhotoController.php
核心变更：删除照片后检查公开照片数量，若为 0 则取消 verified。`,
          },
        );
        console.log(JSON.stringify({
          recoverable: attempt.diagnostics.recoverableEmpty,
          incompleteMutation: attempt.diagnostics.incompleteMutation,
        }));
        """
    )

    assert result.returncode == 0, result.stderr
    assert __import__("json").loads(result.stdout) == {
        "recoverable": True,
        "incompleteMutation": True,
    }


def test_historical_write_does_not_satisfy_current_task_mutation():
    result = run_node(
        r"""
        import { DeepSeekClient } from "./vendor/deepseek-web-api/dist/deepseek/client.js";
        const definitions = [{
          type: "function",
          name: "exec_command",
          parameters: {
            type: "object",
            properties: { cmd: { type: "string" } },
            required: ["cmd"],
          },
        }];
        const client = new DeepSeekClient(
          {},
          {},
          {
            get() {
              return {
                turns: [{
                  role: "user",
                  content: "修改旧任务",
                }, {
                  role: "assistant",
                  content: '<tool_call>{"name":"exec_command","arguments":{"cmd":"sed -i s/old/new/ old.php"}}</tool_call>',
                }, {
                  role: "tool",
                  content: "Process exited with code 0",
                }, {
                  role: "assistant",
                  content: '<tool_call>{"name":"exec_command","arguments":{"cmd":"git diff -- old.php"}}</tool_call>',
                }, {
                  role: "tool",
                  content: "Process exited with code 0\n-old\n+new",
                }, {
                  role: "user",
                  content: "修复 PhotoController 的 verified 状态",
                }, {
                  role: "assistant",
                  content: '<tool_call>{"name":"exec_command","arguments":{"cmd":"grep -n verified PhotoController.php"}}</tool_call>',
                }, {
                  role: "tool",
                  content: "Process exited with code 0\n100: verified",
                }],
              };
            },
          },
          { warn() {} },
        );
        const attempt = client.assessCompletion(
          {
            sessionId: "session-history",
            retry: 0,
            latestUserText: "修复 PhotoController 的 verified 状态",
            requestTurns: [],
            toolDefinitions: definitions,
          },
          {
            diagnostics: {
              recoverableEmpty: false,
              toolCallCount: 0,
              emptyUpstream: false,
            },
            responseText: "已完成修改。修改文件：PhotoController.php",
          },
        );
        console.log(JSON.stringify({
          recoverable: attempt.diagnostics.recoverableEmpty,
          incompleteMutation: attempt.diagnostics.incompleteMutation,
        }));
        """
    )

    assert result.returncode == 0, result.stderr
    assert __import__("json").loads(result.stdout) == {
        "recoverable": True,
        "incompleteMutation": True,
    }


def test_unverified_write_is_forced_to_retry():
    result = run_node(
        r"""
        import { DeepSeekClient } from "./vendor/deepseek-web-api/dist/deepseek/client.js";
        const definitions = [{
          type: "function",
          name: "apply_patch",
          parameters: {
            type: "object",
            properties: { input: { type: "string" } },
            required: ["input"],
          },
        }];
        const client = new DeepSeekClient(
          {},
          {},
          {
            get() {
              return {
                turns: [{
                  role: "user",
                  content: "修复代码",
                }, {
                  role: "assistant",
                  content: '<tool_call>{"name":"apply_patch","arguments":{"input":"*** Begin Patch\\n*** End Patch"}}</tool_call>',
                }, {
                  role: "tool",
                  content: "Done!",
                }],
              };
            },
          },
          { warn() {} },
        );
        const attempt = client.assessCompletion(
          {
            sessionId: "session-unverified",
            retry: 0,
            latestUserText: "修复代码",
            requestTurns: [],
            toolDefinitions: definitions,
          },
          {
            diagnostics: {
              recoverableEmpty: false,
              toolCallCount: 0,
              emptyUpstream: false,
            },
            responseText: "已完成修改。修改文件：target.php",
          },
        );
        console.log(JSON.stringify({
          recoverable: attempt.diagnostics.recoverableEmpty,
          incompleteMutation: attempt.diagnostics.incompleteMutation,
        }));
        """
    )

    assert result.returncode == 0, result.stderr
    assert __import__("json").loads(result.stdout) == {
        "recoverable": True,
        "incompleteMutation": True,
    }


def test_write_and_verification_in_same_command_is_accepted():
    result = run_node(
        r"""
        import { DeepSeekClient } from "./vendor/deepseek-web-api/dist/deepseek/client.js";
        const definitions = [{
          type: "function",
          name: "exec_command",
          parameters: {
            type: "object",
            properties: { cmd: { type: "string" } },
            required: ["cmd"],
          },
        }];
        const client = new DeepSeekClient(
          {},
          {},
          {
            get() {
              return {
                turns: [{
                  role: "user",
                  content: "修复代码",
                }, {
                  role: "assistant",
                  content: '<tool_call>{"name":"exec_command","arguments":{"cmd":"sed -i s/old/new/ target.php && git diff -- target.php"}}</tool_call>',
                }, {
                  role: "tool",
                  content: "Process exited with code 0\n-old\n+new",
                }],
              };
            },
          },
          { warn() { throw new Error("unexpected warning"); } },
        );
        const attempt = client.assessCompletion(
          {
            sessionId: "session-combined",
            retry: 0,
            latestUserText: "修复代码",
            requestTurns: [],
            toolDefinitions: definitions,
          },
          {
            diagnostics: {
              recoverableEmpty: false,
              toolCallCount: 0,
              emptyUpstream: false,
            },
            responseText: "已完成修改。修改文件：target.php",
          },
        );
        console.log(JSON.stringify({
          recoverable: attempt.diagnostics.recoverableEmpty,
          incompleteMutation: attempt.diagnostics.incompleteMutation ?? false,
        }));
        """
    )

    assert result.returncode == 0, result.stderr
    assert __import__("json").loads(result.stdout) == {
        "recoverable": False,
        "incompleteMutation": False,
    }


def test_continue_request_reuses_current_task_verified_write():
    result = run_node(
        r"""
        import { DeepSeekClient } from "./vendor/deepseek-web-api/dist/deepseek/client.js";
        const definitions = [{
          type: "function",
          name: "exec_command",
          parameters: {
            type: "object",
            properties: { cmd: { type: "string" } },
            required: ["cmd"],
          },
        }];
        const client = new DeepSeekClient(
          {},
          {},
          {
            get() {
              return {
                turns: [{
                  role: "user",
                  content: "修复 PhotoController verified 状态",
                }, {
                  role: "assistant",
                  content: '<tool_call>{"name":"exec_command","arguments":{"cmd":"sed -i s/old/new/ PhotoController.php"}}</tool_call>',
                }, {
                  role: "tool",
                  content: "Process exited with code 0",
                }, {
                  role: "assistant",
                  content: '<tool_call>{"name":"exec_command","arguments":{"cmd":"git diff -- PhotoController.php"}}</tool_call>',
                }, {
                  role: "tool",
                  content: "Process exited with code 0\n-old\n+new",
                }, {
                  role: "user",
                  content: "继续",
                }],
              };
            },
          },
          { warn() { throw new Error("unexpected warning"); } },
        );
        const attempt = client.assessCompletion(
          {
            sessionId: "session-continue",
            retry: 0,
            latestUserText: "继续",
            requestTurns: [],
            toolDefinitions: definitions,
          },
          {
            diagnostics: {
              recoverableEmpty: false,
              toolCallCount: 0,
              emptyUpstream: false,
            },
            responseText: "已完成修改。修改文件：PhotoController.php",
          },
        );
        console.log(JSON.stringify({
          recoverable: attempt.diagnostics.recoverableEmpty,
          incompleteMutation: attempt.diagnostics.incompleteMutation ?? false,
        }));
        """
    )

    assert result.returncode == 0, result.stderr
    assert __import__("json").loads(result.stdout) == {
        "recoverable": False,
        "incompleteMutation": False,
    }


def test_legacy_execute_command_xml_maps_to_exec_command():
    result = run_node(
        r"""
        import { resolveToolTurn } from "./vendor/deepseek-web-api/dist/deepseek/toolOutcome.js";
        const tools = [{
          type: "function",
          name: "exec_command",
          parameters: {
            type: "object",
            properties: { cmd: { type: "string" } },
            required: ["cmd"],
          },
        }];
        const outcome = resolveToolTurn(
          `准备搜索。
<execute_command>
<command>find /tmp -name '*.php' -exec grep -l -i 'verified' {} \\; | head -20</command>
</execute_command>`,
          "",
          "legacy",
          "",
          tools,
        );
        const call = outcome.parsed.toolCalls[0];
        console.log(JSON.stringify({
          content: outcome.parsed.content,
          recoverable: outcome.recoverableEmpty,
          name: call?.function.name,
          arguments: call ? JSON.parse(call.function.arguments) : null,
        }));
        """
    )

    assert result.returncode == 0, result.stderr
    assert __import__("json").loads(result.stdout) == {
        "content": "准备搜索。",
        "recoverable": False,
        "name": "exec_command",
        "arguments": {
            "cmd": "find /tmp -name '*.php' -exec grep -l -i 'verified' {} \\; | head -20"
        },
    }


def test_multiple_legacy_execute_command_blocks_are_all_parsed():
    result = run_node(
        r"""
        import { resolveToolTurn } from "./vendor/deepseek-web-api/dist/deepseek/toolOutcome.js";
        const tools = [{
          type: "function",
          name: "exec_command",
          parameters: {
            type: "object",
            properties: { cmd: { type: "string" } },
            required: ["cmd"],
          },
        }];
        const xml = ["one", "two", "three", "four"]
          .map((cmd) => `<execute_command><command>${cmd}</command></execute_command>`)
          .join("\n");
        const outcome = resolveToolTurn(xml, "", "multi", "", tools);
        console.log(JSON.stringify({
          content: outcome.parsed.content,
          recoverable: outcome.recoverableEmpty,
          ids: outcome.parsed.toolCalls.map((call) => call.id),
          commands: outcome.parsed.toolCalls.map((call) => JSON.parse(call.function.arguments).cmd),
        }));
        """
    )

    assert result.returncode == 0, result.stderr
    parsed = __import__("json").loads(result.stdout)
    assert parsed["content"] == ""
    assert parsed["recoverable"] is False
    assert parsed["commands"] == ["one", "two", "three", "four"]
    assert len(parsed["ids"]) == len(set(parsed["ids"])) == 4


def test_call_name_xml_and_mismatched_call_tags_are_recovered():
    result = run_node(
        r"""
        import { parseToolCalls } from "./vendor/deepseek-web-api/dist/deepseek/toolCalls.js";
        const tools = [{
          type: "function",
          name: "exec_command",
          parameters: {
            type: "object",
            properties: { cmd: { type: "string" } },
            required: ["cmd"],
          },
        }];
        const normal = parseToolCalls(
          `<call name="exec_command"><parameter name="cmd">pwd</parameter></call>`,
          "normal",
          tools,
        );
        const mismatched = parseToolCalls(
          `<_call name="exec_command"><parameter name="cmd">git status</parameter></tool_call>`,
          "mismatch",
          tools,
        );
        console.log(JSON.stringify({
          normal: normal.toolCalls.map((call) => JSON.parse(call.function.arguments).cmd),
          mismatched: mismatched.toolCalls.map((call) => JSON.parse(call.function.arguments).cmd),
        }));
        """
    )

    assert result.returncode == 0, result.stderr
    assert __import__("json").loads(result.stdout) == {
        "normal": ["pwd"],
        "mismatched": ["git status"],
    }


def test_malformed_execute_command_is_recoverable_empty():
    result = run_node(
        r"""
        import { resolveToolTurn } from "./vendor/deepseek-web-api/dist/deepseek/toolOutcome.js";
        const tools = [{
          type: "function",
          name: "exec_command",
          parameters: {
            type: "object",
            properties: { cmd: { type: "string" } },
            required: ["cmd"],
          },
        }];
        const outcome = resolveToolTurn(
          `<execute_command><command></command></execute_command>`,
          "",
          "malformed",
          "",
          tools,
        );
        console.log(JSON.stringify({
          content: outcome.parsed.content,
          calls: outcome.parsed.toolCalls.length,
          recoverable: outcome.recoverableEmpty,
        }));
        """
    )

    assert result.returncode == 0, result.stderr
    assert __import__("json").loads(result.stdout) == {
        "content": "",
        "calls": 0,
        "recoverable": True,
    }


def test_codegraph_alias_only_maps_when_concrete_tool_is_available():
    result = run_node(
        r"""
        import { parseToolCalls } from "./vendor/deepseek-web-api/dist/deepseek/toolCalls.js";
        const concrete = [{
          type: "function",
          name: "codegraph_explore",
          parameters: {
            type: "object",
            properties: {
              projectPath: { type: "string" },
              query: { type: "string" },
            },
            required: ["projectPath", "query"],
          },
        }];
        const namespaceOnly = [{
          type: "function",
          name: "mcp__codegraph__",
          parameters: {
            type: "object",
            properties: {},
          },
        }];
        const text = `<tool_call>{"name":"mcp__codegraph","arguments":{"projectPath":"/tmp","query":"User verified"}}</tool_call>`;
        const mapped = parseToolCalls(text, "mapped", concrete);
        const rejected = parseToolCalls(text, "rejected", namespaceOnly);
        console.log(JSON.stringify({
          mapped: mapped.toolCalls.map((call) => call.function.name),
          rejected: rejected.toolCalls.length,
        }));
        """
    )

    assert result.returncode == 0, result.stderr
    assert __import__("json").loads(result.stdout) == {
        "mapped": ["codegraph_explore"],
        "rejected": 0,
    }


def test_codegraph_alias_falls_back_to_codegraph_cli_when_exec_is_available():
    result = run_node(
        r"""
        import { parseToolCalls } from "./vendor/deepseek-web-api/dist/deepseek/toolCalls.js";
        const tools = [{
          type: "function",
          name: "exec_command",
          parameters: {
            type: "object",
            properties: {
              cmd: { type: "string" },
              workdir: { type: "string" },
            },
            required: ["cmd"],
          },
        }];
        const parsed = parseToolCalls(
          `<tool_call>{"name":"mcp__codegraph","arguments":{"input":"explore censor verified status","projectPath":"/opt/codes/ms/de4-web"}}</tool_call>`,
          "codegraph-cli",
          tools,
        );
        const call = parsed.toolCalls[0];
        console.log(JSON.stringify({
          name: call?.function.name,
          arguments: call ? JSON.parse(call.function.arguments) : null,
        }));
        """
    )

    assert result.returncode == 0, result.stderr
    assert __import__("json").loads(result.stdout) == {
        "name": "exec_command",
        "arguments": {
            "cmd": "codegraph explore 'censor verified status'",
            "workdir": "/opt/codes/ms/de4-web",
        },
    }
