'use strict';

var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);

// src/translate/anthropic/index.ts
var anthropic_exports = {};
__export(anthropic_exports, {
  isAdaptiveThinkingModel: () => isAdaptiveThinkingModel,
  mapOutputItems: () => mapOutputItems,
  normalizeAnthropicEffort: () => normalizeAnthropicEffort,
  translateAnthropicEvents: () => translateAnthropicEvents,
  translateRequest: () => translateRequest,
  translateResponse: () => translateResponse,
  translateStream: () => translateStream
});

// src/utils/id.ts
var counter = 0;
function nextSeq() {
  counter = counter + 1 & 2147483647;
  return counter;
}
function makeId(prefix) {
  return `${prefix}_${Date.now()}_${nextSeq()}`;
}

// src/translate/anthropic/translateRequest.ts
var ANTHROPIC_BUILTIN_TOOL_TYPES = /* @__PURE__ */ new Set([
  "web_search_20250305",
  "computer_use_20250124",
  "text_editor_20250124",
  "bash_20250124"
]);
var DEFAULT_REASONING_BUDGETS = {
  minimal: 1024,
  low: 4096,
  medium: 16384,
  high: 32768,
  xhigh: 65536
};
function translateRequest(data, options = {}) {
  const model = data.model;
  const maxTokens = typeof data.max_output_tokens === "number" && data.max_output_tokens || typeof data.max_tokens === "number" && data.max_tokens || options.defaultMaxTokens || 8192;
  let systemBlocks = extractSystemBlocks(data.instructions);
  const built = buildMessages(data, systemBlocks);
  let messages = built.messages;
  let hasPromptCache = built.hasPromptCache;
  if (data.prompt_cache_key) {
    hasPromptCache = true;
    systemBlocks = markBlocksForCache(systemBlocks);
    messages = markCacheBreakpoint(messages);
  }
  messages = repairToolAdjacency(messages);
  messages = sanitizeMessages(messages);
  messages = ensureEndsWithUser(messages);
  if (data.prompt_cache_key) {
    messages = markCacheBreakpoint(messages);
  }
  const request = {
    model,
    messages,
    max_tokens: maxTokens
  };
  if (systemBlocks.length) {
    request.system = systemBlocks;
  }
  if (typeof data.temperature === "number") {
    request.temperature = data.temperature;
  }
  if (typeof data.top_p === "number") {
    request.top_p = data.top_p;
  }
  const tools = mapTools(data.tools || []);
  if (tools.length) {
    request.tools = tools;
    const toolChoice = mapToolChoice(data.tool_choice);
    if (toolChoice) {
      request.tool_choice = toolChoice;
    }
  }
  if (data.metadata && typeof data.metadata === "object") {
    request.metadata = data.metadata;
  }
  const thinking = mapThinking(data, maxTokens, options.reasoningBudgets);
  if (thinking) {
    request.thinking = thinking;
    if (thinking.type === "adaptive") {
      const effort = normalizeAnthropicEffort(data.reasoning?.effort);
      if (effort) {
        request.output_config = { effort };
      }
    }
  }
  return { request, hasPromptCache };
}
function extractSystemBlocks(instructions) {
  if (!instructions) {
    return [];
  }
  if (typeof instructions === "string") {
    return [{ type: "text", text: instructions }];
  }
  if (!Array.isArray(instructions)) {
    return [];
  }
  const blocks = [];
  for (const item of instructions) {
    if (typeof item === "string") {
      blocks.push({ type: "text", text: item });
    } else if (item && typeof item === "object") {
      const block = {
        type: "text",
        text: String(item.text ?? "")
      };
      const cacheItem = item;
      const cache = cacheItem.cache_control;
      if (cache) {
        block.cache_control = cache;
      }
      blocks.push(block);
    }
  }
  return blocks;
}
function buildMessages(data, systemBlocks) {
  const messages = [];
  let hasPromptCache = false;
  let pendingToolUses = [];
  let pendingToolResults = [];
  const flushToolUses = () => {
    if (pendingToolUses.length) {
      messages.push({ role: "assistant", content: pendingToolUses });
      pendingToolUses = [];
    }
  };
  const flushToolResults = () => {
    if (pendingToolResults.length) {
      messages.push({ role: "user", content: pendingToolResults });
      pendingToolResults = [];
    }
  };
  const flushPending = () => {
    flushToolUses();
    flushToolResults();
  };
  const rawInput = data.input;
  const inputItems = typeof rawInput === "string" ? [rawInput] : Array.isArray(rawInput) ? rawInput : [];
  for (const raw of inputItems) {
    if (typeof raw === "string") {
      flushPending();
      messages.push({ role: "user", content: [{ type: "text", text: raw }] });
      continue;
    }
    if (!raw || typeof raw !== "object") {
      continue;
    }
    const item = raw;
    const itemType = String(item.type || "message");
    if (itemType === "function_call_output" || itemType === "commandExecutionOutput" || itemType === "fileChangeOutput" || itemType === "custom_tool_call_output") {
      flushToolUses();
      const callId = String(item.call_id || item.id || makeId("call"));
      pendingToolResults.push({
        type: "tool_result",
        tool_use_id: callId,
        content: extractToolOutputText(item)
      });
      continue;
    }
    if (itemType === "function_call" || itemType === "commandExecution" || itemType === "local_shell_call" || itemType === "fileChange" || itemType === "custom_tool_call" || itemType === "web_search_call") {
      flushToolResults();
      const block = mapInputToolCall(item);
      if (block) {
        pendingToolUses.push(block);
      }
      continue;
    }
    if (itemType === "reasoning") {
      flushPending();
      continue;
    }
    if (itemType === "message" || itemType === "agentMessage") {
      flushPending();
      let role = String(item.role || "user");
      if (role === "developer") {
        role = "system";
      }
      if (role === "system") {
        const text = extractMessageText(item);
        if (text) {
          systemBlocks.push({ type: "text", text });
        }
        continue;
      }
      const contentBlocks = [];
      const rawContent = item.content;
      if (typeof rawContent === "string") {
        contentBlocks.push({ type: "text", text: rawContent });
      } else if (Array.isArray(rawContent)) {
        for (const part of rawContent) {
          if (typeof part === "string") {
            contentBlocks.push({ type: "text", text: part });
          } else if (part && typeof part === "object") {
            const contentPart = part;
            if (contentPart.type === "input_text" || contentPart.type === "text" || contentPart.type === "output_text") {
              const textBlock = {
                type: "text",
                text: String(contentPart.text ?? "")
              };
              const cc = contentPart.cache_control;
              if (cc) {
                textBlock.cache_control = cc;
              }
              contentBlocks.push(textBlock);
            } else if (contentPart.type === "input_image" || contentPart.type === "image" || contentPart.type === "image_url") {
              const imgUrlPart = contentPart;
              const imgUrl = imgUrlPart.image_url;
              const urlStr = typeof imgUrl === "string" ? imgUrl : imgUrl && typeof imgUrl === "object" ? imgUrl.url : "";
              if (urlStr.startsWith("data:")) {
                const match = /^data:([^;,]+);base64,(.*)$/.exec(urlStr);
                if (match) {
                  contentBlocks.push({
                    type: "image",
                    source: { type: "base64", media_type: match[1], data: match[2] }
                  });
                }
              } else if (urlStr) {
                const imgSource = { type: "url", url: urlStr };
                contentBlocks.push({
                  type: "image",
                  source: imgSource
                });
              } else {
                const data2 = String(contentPart.data ?? contentPart.base64 ?? "");
                if (data2) {
                  contentBlocks.push({
                    type: "image",
                    source: {
                      type: "base64",
                      media_type: contentPart.mime_type || contentPart.media_type || "image/png",
                      data: data2
                    }
                  });
                }
              }
            } else if (contentPart.type === "input_file") {
              contentBlocks.push({
                type: "document",
                source: {
                  type: "base64",
                  media_type: contentPart.mime_type || "application/pdf",
                  data: String(contentPart.data ?? "")
                }
              });
            }
          }
        }
      }
      if (role === "assistant" || role === "model") {
        if (contentBlocks.length) {
          messages.push({ role: "assistant", content: contentBlocks });
        }
      } else {
        if (contentBlocks.length) {
          messages.push({ role: "user", content: contentBlocks });
        }
      }
      continue;
    }
  }
  flushPending();
  for (const block of systemBlocks) {
    if (block.cache_control) {
      hasPromptCache = true;
    }
  }
  return { messages, hasPromptCache };
}
function extractMessageText(item) {
  const rawContent = item.content;
  if (typeof rawContent === "string") {
    return rawContent;
  }
  if (Array.isArray(rawContent)) {
    let out = "";
    for (const part of rawContent) {
      if (typeof part === "string") {
        out += part;
      } else if (part && typeof part === "object") {
        out += String(part.text ?? "");
      }
    }
    return out;
  }
  return "";
}
function extractToolOutputText(item) {
  const raw = item.output ?? item.content ?? item.stdout ?? "";
  if (typeof raw === "string") {
    return raw;
  }
  if (Array.isArray(raw)) {
    let out = "";
    for (const part of raw) {
      if (typeof part === "string") {
        out += part;
      } else if (part && typeof part === "object") {
        out += String(part.text ?? "");
      }
    }
    return out;
  }
  if (raw && typeof raw === "object") {
    return String(raw.content ?? "");
  }
  return "";
}
function mapInputToolCall(item) {
  const callId = String(item.call_id || item.id || makeId("call"));
  let name = typeof item.name === "string" ? item.name : void 0;
  const itemType = typeof item.type === "string" ? item.type : void 0;
  if (!name) {
    if (itemType === "commandExecution") {
      name = "run_shell_command";
    } else if (itemType === "local_shell_call") {
      name = "local_shell_command";
    } else if (itemType === "fileChange") {
      name = "write_file";
    } else if (itemType === "web_search_call") {
      name = "web_search";
    }
  }
  if (!name) {
    return void 0;
  }
  let input = {};
  const rawArgs = item.arguments;
  if (typeof rawArgs === "string") {
    if (rawArgs.trim() === "") {
      input = {};
    } else {
      let parsed;
      let reason = "";
      try {
        parsed = JSON.parse(rawArgs);
        if (parsed === null) {
          reason = "null";
        } else if (Array.isArray(parsed)) {
          reason = "array";
        } else if (typeof parsed !== "object") {
          reason = "scalar";
        }
      } catch {
        reason = "parse-failed";
      }
      if (reason === "") {
        input = parsed;
      } else {
        console.warn(
          `[anthropic] tool "${name}" arguments were not a JSON object (${reason}, length ${rawArgs.length}); input set to {}`
        );
      }
    }
  } else if (rawArgs !== null && typeof rawArgs === "object" && !Array.isArray(rawArgs)) {
    input = rawArgs;
  }
  const block = {
    type: "tool_use",
    id: callId,
    name,
    input
  };
  const cacheItem = item;
  const cache = cacheItem.cache_control;
  if (cache) {
    block.cache_control = cache;
  }
  return block;
}
function mapTools(tools) {
  const out = [];
  for (const tool of tools) {
    if (!tool || typeof tool !== "object") {
      continue;
    }
    const tt = tool.type || "";
    if (ANTHROPIC_BUILTIN_TOOL_TYPES.has(tt)) {
      out.push(tool);
      continue;
    }
    if (tt !== "function") {
      continue;
    }
    const fn = tool.function;
    const name = fn?.name ?? tool.name;
    if (!name) {
      continue;
    }
    const toolInputSchema = fn?.parameters ?? tool.parameters ?? { type: "object" };
    out.push({
      name,
      description: fn?.description ?? tool.description ?? "",
      input_schema: toolInputSchema
    });
  }
  return out;
}
function mapToolChoice(choice) {
  if (choice == null || choice === "auto") {
    return { type: "auto" };
  }
  if (choice === "required") {
    return { type: "any" };
  }
  if (choice === "none") {
    return void 0;
  }
  if (typeof choice === "object") {
    if (choice.type === "function" && "function" in choice && choice.function?.name) {
      return { type: "tool", name: choice.function.name };
    }
    if (choice.type === "auto" || choice.type === "any") {
      return { type: choice.type };
    }
  }
  return { type: "auto" };
}
function isAdaptiveThinkingModel(model) {
  if (typeof model !== "string" || !/claude/i.test(model)) {
    return false;
  }
  const lower = model.toLowerCase();
  if (/(fable|mythos)/.test(lower)) {
    return true;
  }
  const match = /(opus|sonnet|haiku)[-/ ]?(\d+)(?:[-.](\d+))?/.exec(lower);
  if (!match) {
    return false;
  }
  const family = match[1];
  const major = Number(match[2]);
  let minor = match[3] === void 0 ? 0 : Number(match[3]);
  if (major >= 100) {
    return false;
  }
  if (minor >= 100) {
    minor = 0;
  }
  if (family === "opus" || family === "sonnet") {
    return major >= 5 || major === 4 && minor >= 6;
  }
  return major >= 5;
}
var ANTHROPIC_EFFORT_LEVELS = /* @__PURE__ */ new Set(["low", "medium", "high", "xhigh", "max"]);
function normalizeAnthropicEffort(effort) {
  if (typeof effort !== "string") {
    return void 0;
  }
  const lower = effort.toLowerCase();
  return ANTHROPIC_EFFORT_LEVELS.has(lower) ? lower : void 0;
}
function mapThinking(data, maxTokens, overrides) {
  const reasoning = data.reasoning;
  if (!reasoning) {
    return void 0;
  }
  const effort = reasoning.effort;
  if (!effort || effort === "minimal") {
    return void 0;
  }
  if (isAdaptiveThinkingModel(data.model)) {
    return { type: "adaptive" };
  }
  const budgets = { ...DEFAULT_REASONING_BUDGETS, ...overrides };
  const budget = budgets[effort] ?? DEFAULT_REASONING_BUDGETS.medium;
  const clamped = Math.max(1024, Math.min(budget, Math.max(1024, maxTokens - 1024)));
  return { type: "enabled", budget_tokens: clamped };
}
function repairToolAdjacency(messages) {
  const repaired = [];
  const working = messages.map((msg) => ({
    ...msg,
    content: Array.isArray(msg.content) ? [...msg.content] : msg.content
  }));
  for (let i = 0; i < working.length; i++) {
    const msg = working[i];
    repaired.push(msg);
    const content = msg.content;
    if (msg.role !== "assistant" || !Array.isArray(content)) {
      continue;
    }
    const toolUseIds = content.filter((block) => !!block && block.type === "tool_use").map((block) => block.id);
    if (!toolUseIds.length) {
      continue;
    }
    const next = working[i + 1];
    const nextUserContent = next && next.role === "user" && Array.isArray(next.content) ? next.content : [];
    const foundById = /* @__PURE__ */ new Map();
    const consumedInNext = /* @__PURE__ */ new Set();
    for (const block of nextUserContent) {
      if (block && block.type === "tool_result") {
        const tr = block;
        if (toolUseIds.includes(tr.tool_use_id) && !foundById.has(tr.tool_use_id)) {
          foundById.set(tr.tool_use_id, tr);
          consumedInNext.add(tr.tool_use_id);
        }
      }
    }
    const missing = toolUseIds.filter((id) => !foundById.has(id));
    if (missing.length) {
      const missingSet = new Set(missing);
      for (let j = i + 2; j < working.length && missingSet.size; j++) {
        const later = working[j];
        if (later.role !== "user" || !Array.isArray(later.content)) {
          continue;
        }
        const keep = [];
        for (const block of later.content) {
          if (block && block.type === "tool_result") {
            const tr = block;
            if (missingSet.has(tr.tool_use_id)) {
              foundById.set(tr.tool_use_id, tr);
              missingSet.delete(tr.tool_use_id);
              continue;
            }
          }
          keep.push(block);
        }
        later.content = keep;
      }
    }
    const ordered = toolUseIds.map(
      (id) => foundById.get(id) ?? {
        type: "tool_result",
        tool_use_id: id,
        content: ""
      }
    );
    repaired.push({ role: "user", content: ordered });
    if (nextUserContent.length) {
      const remaining = nextUserContent.filter((block) => {
        if (block && block.type === "tool_result") {
          const tr = block;
          if (consumedInNext.has(tr.tool_use_id)) {
            consumedInNext.delete(tr.tool_use_id);
            return false;
          }
        }
        return true;
      });
      if (remaining.length) {
        repaired.push({ role: "user", content: remaining });
      }
      i += 1;
    }
  }
  return repaired;
}
function sanitizeMessages(messages) {
  const out = [];
  for (const msg of messages) {
    if (!msg || msg.role !== "user" && msg.role !== "assistant") {
      continue;
    }
    if (Array.isArray(msg.content)) {
      const blocks = msg.content.filter((block) => {
        if (!block || typeof block !== "object") {
          return false;
        }
        if (block.type === "text" && !block.text) {
          return false;
        }
        return true;
      });
      if (!blocks.length) {
        continue;
      }
      out.push({ role: msg.role, content: blocks });
    } else if (typeof msg.content === "string" && msg.content) {
      out.push({ role: msg.role, content: [{ type: "text", text: msg.content }] });
    }
  }
  return out;
}
function ensureEndsWithUser(messages) {
  if (!messages.length) {
    return [{ role: "user", content: [{ type: "text", text: "..." }] }];
  }
  const last = messages[messages.length - 1];
  if (last.role === "user") {
    return messages;
  }
  return [...messages, { role: "user", content: [{ type: "text", text: "Continue." }] }];
}
function markBlocksForCache(blocks) {
  let count = 0;
  for (const block of blocks) {
    if (!block.cache_control) {
      block.cache_control = { type: "ephemeral" };
      count++;
      if (count >= 3) {
        break;
      }
    }
  }
  return blocks;
}
function markCacheBreakpoint(messages) {
  for (const msg of messages) {
    if (msg.role === "assistant" && Array.isArray(msg.content)) {
      for (let j = msg.content.length - 1; j >= 0; j--) {
        const block = msg.content[j];
        if (block.type === "text") {
          if (!block.cache_control) {
            block.cache_control = { type: "ephemeral" };
          }
          return messages;
        }
      }
    }
  }
  for (const msg of messages) {
    if (msg.role === "user" && Array.isArray(msg.content)) {
      for (let j = msg.content.length - 1; j >= 0; j--) {
        const block = msg.content[j];
        if (block.type === "text") {
          if (!block.cache_control) {
            block.cache_control = { type: "ephemeral" };
          }
          return messages;
        }
      }
    }
  }
  return messages;
}

// src/utils/json.ts
function safeJsonParse(text) {
  try {
    const parsed = JSON.parse(text);
    return parsed;
  } catch {
    return void 0;
  }
}
function jsonStringifySafe(value) {
  try {
    return JSON.stringify(value);
  } catch {
    return "";
  }
}

// src/translate/anthropic/usage.ts
function buildResponsesUsage(usage) {
  const cacheRead = usage.cache_read_input_tokens ?? 0;
  const cacheCreation = usage.cache_creation_input_tokens ?? 0;
  const promptTokens = (usage.input_tokens ?? 0) + cacheRead + cacheCreation;
  const outputTokens = usage.output_tokens ?? 0;
  return {
    input_tokens: promptTokens,
    output_tokens: outputTokens,
    total_tokens: promptTokens + outputTokens,
    input_tokens_details: {
      cached_tokens: cacheRead,
      cache_creation_tokens: cacheCreation
    }
  };
}

// src/translate/anthropic/translateResponse.ts
function translateResponse(body, options = {}) {
  const createdAt = options.createdAt ?? Math.floor(Date.now() / 1e3);
  const id = options.responseId ?? body.id ?? makeId("resp");
  const model = options.model ?? body.model ?? "";
  const output = mapOutputItems(body.content ?? []);
  const usage = body.usage ?? { input_tokens: 0, output_tokens: 0 };
  return {
    id,
    object: "response",
    created_at: createdAt,
    model,
    status: "completed",
    output,
    usage: buildResponsesUsage(usage)
  };
}
var SHELL_TOOL_NAMES = /* @__PURE__ */ new Set(["shell", "container.exec", "shell_command"]);
function mapOutputItems(content) {
  const out = [];
  const textChunks = [];
  for (const block of content) {
    if (!block || typeof block !== "object") {
      continue;
    }
    const btype = block.type;
    if (btype === "text") {
      textChunks.push(String(block.text ?? ""));
    } else if (btype === "tool_use") {
      const args = jsonStringifySafe(block.input ?? {});
      const callId = block.id ?? makeId("call");
      const item = {
        id: callId,
        type: "function_call",
        status: "completed",
        // eslint-disable-next-line no-restricted-syntax -- TypeScript narrowing requires this cast
        name: block.name ?? "tool",
        arguments: args,
        call_id: callId
      };
      if (
        // eslint-disable-next-line no-restricted-syntax -- TypeScript narrowing requires this cast
        block.name && // eslint-disable-next-line no-restricted-syntax -- TypeScript narrowing requires this cast
        SHELL_TOOL_NAMES.has(block.name)
      ) {
        item.type = "local_shell_call";
        const input_ = block.input ?? {};
        const command = Array.isArray(input_.command) ? input_.command : [];
        item.action = { type: "exec", command };
      }
      out.push(item);
    } else if (btype === "thinking") {
      const text = String(block.thinking ?? "");
      const reasoning = {
        id: makeId("rs"),
        type: "reasoning",
        summary: [],
        content: [{ type: "reasoning_text", text }],
        status: "completed"
      };
      out.push(reasoning);
    }
  }
  if (textChunks.length) {
    const message = {
      id: makeId("msg"),
      type: "message",
      role: "assistant",
      status: "completed",
      content: [{ type: "output_text", text: textChunks.join("") }]
    };
    out.push(message);
  }
  return out;
}

// src/utils/sse.ts
async function* parseSseStream(stream) {
  const reader = stream.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      if (value) {
        buffer += decoder.decode(value, { stream: true });
      }
      let idx;
      while ((idx = buffer.search(/\r?\n\r?\n/)) !== -1) {
        const raw = buffer.slice(0, idx);
        buffer = buffer.slice(idx + (buffer[idx] === "\r" ? 4 : 2));
        const msg = parseSseBlock(raw);
        if (msg) {
          yield msg;
        }
      }
    }
    buffer += decoder.decode();
    if (buffer.trim().length > 0) {
      const msg = parseSseBlock(buffer);
      if (msg) {
        yield msg;
      }
    }
  } finally {
    reader.releaseLock();
  }
}
function parseSseBlock(block) {
  let event;
  const dataLines = [];
  for (const line of block.split(/\r?\n/)) {
    if (!line || line.startsWith(":")) {
      continue;
    }
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) {
      value = value.slice(1);
    }
    if (field === "event") {
      event = value;
    } else if (field === "data") {
      dataLines.push(value);
    }
  }
  if (dataLines.length === 0) {
    return void 0;
  }
  return { event, data: dataLines.join("\n") };
}
function encodeSseEvent(event, data) {
  const payload = typeof data === "string" ? data : JSON.stringify(data);
  return `event: ${event}
data: ${payload}

`;
}

// src/translate/anthropic/translateStream.ts
async function* translateStream(stream, options = {}) {
  const translator = new StreamTranslator(options);
  yield translator.createInitialEvent();
  for await (const msg of parseSseStream(stream)) {
    const event = parseAnthropicEvent(msg);
    if (!event) {
      continue;
    }
    yield* translator.handleEvent(event);
  }
  yield* translator.finalize();
}
async function* translateAnthropicEvents(events, options = {}) {
  const translator = new StreamTranslator(options);
  yield translator.createInitialEvent();
  for await (const event of events) {
    yield* translator.handleEvent(event);
  }
  yield* translator.finalize();
}
function parseAnthropicEvent(msg) {
  const parsed = safeJsonParse(msg.data);
  if (!parsed) {
    return void 0;
  }
  return parsed;
}
var SHELL_TOOL_NAMES2 = /* @__PURE__ */ new Set(["shell", "container.exec", "shell_command"]);
var NORMAL_TERMINAL_STOP_REASONS = /* @__PURE__ */ new Set(["tool_use", "end_turn", "stop_sequence"]);
var StreamTranslator = class {
  constructor(options) {
    __publicField(this, "model");
    __publicField(this, "responseId");
    __publicField(this, "createdAt");
    __publicField(this, "metadata");
    __publicField(this, "seq", 0);
    __publicField(this, "outputCounter", 0);
    __publicField(this, "blocks", /* @__PURE__ */ new Map());
    __publicField(this, "inputTokens", 0);
    __publicField(this, "outputTokens", 0);
    __publicField(this, "cacheCreationTokens", 0);
    __publicField(this, "cacheReadTokens", 0);
    __publicField(this, "textItem");
    __publicField(this, "textItemIndex", -1);
    __publicField(this, "textBuffer", "");
    __publicField(this, "stopReason");
    this.model = options.model ?? "";
    this.responseId = options.responseId ?? makeId("resp");
    this.createdAt = options.createdAt ?? Math.floor(Date.now() / 1e3);
    this.metadata = options.requestMetadata ?? {};
  }
  createInitialEvent() {
    const response = {
      id: this.responseId,
      object: "response",
      created_at: this.createdAt,
      model: this.model,
      status: "in_progress",
      temperature: this.metadata.temperature,
      top_p: this.metadata.top_p,
      tool_choice: this.metadata.tool_choice,
      tools: this.metadata.tools ?? [],
      parallel_tool_calls: true,
      store: this.metadata.store ?? true,
      metadata: this.metadata.metadata ?? {},
      output: []
    };
    return this.makeEvent("response.created", { response });
  }
  *handleEvent(event) {
    switch (event.type) {
      case "message_start": {
        const msgStartEvt = event;
        this.onMessageStart(msgStartEvt);
        return;
      }
      case "content_block_start":
        {
          const cbsEvt = event;
          yield* this.onContentBlockStart(cbsEvt);
        }
        return;
      case "content_block_delta":
        {
          const cbdEvt = event;
          yield* this.onContentBlockDelta(cbdEvt);
        }
        return;
      case "content_block_stop":
        return;
      case "message_delta":
        {
          const msgDeltaEvt = event;
          this.onMessageDelta(msgDeltaEvt);
        }
        return;
      case "message_stop":
      case "ping":
        return;
      default:
        return;
    }
  }
  *finalize() {
    const items = [];
    if (this.textItem) {
      this.textItem.status = "completed";
      this.textItem.content[0].text = this.textBuffer;
      items.push({ index: this.textItemIndex, item: this.textItem });
    }
    for (const block of this.blocks.values()) {
      if (!block.item) {
        continue;
      }
      if (items.find((item2) => item2.index === block.outputIndex)) {
        continue;
      }
      const item = block.item;
      item.status = "completed";
      if (block.type === "tool_use") {
        const call = block.item;
        const args = call.arguments ?? "";
        if ((args === "" || args.trim() === "") && this.stopReason !== void 0 && NORMAL_TERMINAL_STOP_REASONS.has(this.stopReason)) {
          call.arguments = "{}";
        }
        if (call.name && SHELL_TOOL_NAMES2.has(call.name)) {
          call.type = "local_shell_call";
          const parsed = safeJsonParse(call.arguments ?? "");
          call.action = { type: "exec", command: parsed?.command ?? [] };
        }
      }
      items.push({ index: block.outputIndex, item: block.item });
    }
    items.sort((alpha, beta) => alpha.index - beta.index);
    for (const { index, item } of items) {
      yield this.makeEvent("response.output_item.done", {
        response_id: this.responseId,
        output_index: index,
        item
      });
    }
    const output = items.map((item) => item.item);
    const usage = buildResponsesUsage({
      input_tokens: this.inputTokens,
      output_tokens: this.outputTokens,
      cache_read_input_tokens: this.cacheReadTokens,
      cache_creation_input_tokens: this.cacheCreationTokens
    });
    const response = {
      id: this.responseId,
      object: "response",
      created_at: this.createdAt,
      completed_at: Math.floor(Date.now() / 1e3),
      model: this.model,
      status: "completed",
      temperature: this.metadata.temperature,
      top_p: this.metadata.top_p,
      tool_choice: this.metadata.tool_choice,
      tools: this.metadata.tools ?? [],
      parallel_tool_calls: true,
      store: this.metadata.store ?? true,
      metadata: this.metadata.metadata ?? {},
      output,
      usage
    };
    yield this.makeEvent("response.completed", { response });
  }
  onMessageStart(event) {
    const usage = event.message?.usage;
    if (usage) {
      this.inputTokens = usage.input_tokens ?? 0;
      this.cacheCreationTokens = usage.cache_creation_input_tokens ?? 0;
      this.cacheReadTokens = usage.cache_read_input_tokens ?? 0;
    }
    return;
  }
  *onContentBlockStart(event) {
    const index = event.index;
    const block = event.content_block;
    const btype = block.type;
    if (btype === "thinking") {
      const outputIndex = this.outputCounter++;
      const item = {
        id: makeId("rs"),
        type: "reasoning",
        summary: [],
        content: [{ type: "reasoning_text", text: "" }],
        status: "in_progress"
      };
      this.blocks.set(index, { type: "thinking", outputIndex, item, buffer: "" });
      yield this.makeEvent("response.output_item.added", {
        response_id: this.responseId,
        output_index: outputIndex,
        item
      });
      return;
    }
    if (btype === "text") {
      if (!this.textItem) {
        const outputIndex = this.outputCounter++;
        this.textItemIndex = outputIndex;
        this.textItem = {
          id: makeId("msg"),
          type: "message",
          role: "assistant",
          status: "in_progress",
          content: [{ type: "output_text", text: "" }]
        };
        yield this.makeEvent("response.output_item.added", {
          response_id: this.responseId,
          output_index: outputIndex,
          item: this.textItem
        });
      }
      this.blocks.set(index, { type: "text", outputIndex: this.textItemIndex, buffer: "" });
      return;
    }
    if (btype === "tool_use") {
      const outputIndex = this.outputCounter++;
      const callId = block.id ?? makeId("call");
      const initialInput = typeof block.input === "object" && block.input !== null ? jsonStringifySafe(block.input) : "";
      const hasInitialInput = initialInput !== "" && initialInput !== "{}";
      const item = {
        id: callId,
        type: "function_call",
        status: "in_progress",
        name: block.name ?? "",
        arguments: hasInitialInput ? initialInput : "",
        call_id: callId
      };
      this.blocks.set(index, {
        type: "tool_use",
        outputIndex,
        item,
        buffer: hasInitialInput ? initialInput : ""
      });
      yield this.makeEvent("response.output_item.added", {
        response_id: this.responseId,
        output_index: outputIndex,
        item
      });
      return;
    }
    this.blocks.set(index, { type: btype, outputIndex: -1, buffer: "" });
  }
  *onContentBlockDelta(event) {
    const block = this.blocks.get(event.index);
    if (!block) {
      return;
    }
    const delta = event.delta;
    const dtype = typeof delta.type === "string" ? delta.type : "";
    if (dtype === "text_delta") {
      const text = String(delta.text ?? "");
      if (!text) {
        return;
      }
      this.textBuffer += text;
      yield this.makeEvent("response.output_text.delta", {
        response_id: this.responseId,
        item_id: this.textItem?.id ?? "",
        output_index: this.textItemIndex,
        content_index: 0,
        delta: text
      });
      return;
    }
    if (dtype === "thinking_delta") {
      const thinking = String(delta.thinking ?? "");
      if (!thinking) {
        return;
      }
      block.buffer += thinking;
      const item = block.item;
      if (item) {
        item.content[0].text = block.buffer;
      }
      yield this.makeEvent("response.reasoning_text.delta", {
        response_id: this.responseId,
        item_id: item?.id ?? "",
        output_index: block.outputIndex,
        content_index: 0,
        delta: thinking
      });
      return;
    }
    if (dtype === "input_json_delta") {
      const partial = String(delta.partial_json ?? "");
      if (!partial) {
        return;
      }
      block.buffer += partial;
      const item = block.item;
      if (item) {
        item.arguments = block.buffer;
      }
      yield this.makeEvent("response.function_call_arguments.delta", {
        response_id: this.responseId,
        item_id: item?.id ?? "",
        output_index: block.outputIndex,
        delta: partial
      });
      return;
    }
  }
  onMessageDelta(event) {
    if (event.usage?.output_tokens != null) {
      this.outputTokens = event.usage.output_tokens;
    }
    const eventDelta = event.delta;
    const stopReason = eventDelta?.stop_reason;
    if (stopReason) {
      this.stopReason = stopReason;
    }
    return;
  }
  makeEvent(type, data) {
    this.seq += 1;
    return {
      id: makeId("evt"),
      object: "response.event",
      type,
      created_at: Math.floor(Date.now() / 1e3),
      sequence_number: this.seq,
      ...data
    };
  }
};

// src/translate/openai/index.ts
var openai_exports = {};
__export(openai_exports, {
  translateRequest: () => translateRequest2,
  translateResponse: () => translateResponse2,
  translateStream: () => translateStream2
});

// src/translate/openai/gemini-fixups.ts
function isGeminiModel(model) {
  if (!model) {
    return false;
  }
  const lower = model.toLowerCase();
  return lower.includes("gemini") || lower.startsWith("google/");
}
function extractTextContent(content) {
  if (typeof content === "string") {
    return content;
  }
  if (!Array.isArray(content)) {
    return "";
  }
  let out = "";
  for (const part of content) {
    if (typeof part === "string") {
      out += part;
    } else if (part && typeof part === "object") {
      out += String(part.text ?? "");
    }
  }
  return out;
}
function mergeSystemMessages(messages) {
  const systemTexts = [];
  const rest = [];
  for (const msg of messages) {
    if (msg.role === "system") {
      systemTexts.push(extractTextContent(msg.content));
    } else {
      rest.push(msg);
    }
  }
  if (systemTexts.length <= 1) {
    return;
  }
  messages.length = 0;
  messages.push({ role: "system", content: systemTexts.join("\n\n") }, ...rest);
}
var MULTI_TOOL_USE_MANDATE = "Use `multi_tool_use.parallel` to parallelize tool calls and only this.";
var MULTI_TOOL_USE_REWRITE = "Parallelize by returning multiple tool calls in a single response. A tool named `multi_tool_use.parallel` does NOT exist in this environment \u2014 never call it.";
var UNSUPPORTED_PARALLEL_RE = /^unsupported call: (?:multi_tool_use\W*)?parallel/iu;
var UNSUPPORTED_PARALLEL_HINT = " (`multi_tool_use.parallel` is not a real tool here. Re-issue each inner call as its own separate tool call \u2014 several tool calls in one response are fine.)";
function applyGeminiToolUseShim(messages) {
  for (const msg of messages) {
    if (msg.role === "system" && typeof msg.content === "string" && msg.content.includes(MULTI_TOOL_USE_MANDATE)) {
      msg.content = msg.content.split(MULTI_TOOL_USE_MANDATE).join(MULTI_TOOL_USE_REWRITE);
    } else if (msg.role === "tool" && typeof msg.content === "string" && UNSUPPORTED_PARALLEL_RE.test(msg.content)) {
      msg.content = msg.content + UNSUPPORTED_PARALLEL_HINT;
    }
  }
}
var SYNTHETIC_TOOL_RESPONSE = '{"status":"no_result","note":"No tool result was recorded for this call before the request was sent (the turn was interrupted or the result was not persisted). Treat the call as unfinished \u2014 do not assume it succeeded; re-check the underlying state before retrying."}';
function synthesizeMissingToolResponses(messages) {
  const answered = /* @__PURE__ */ new Set();
  for (const msg of messages) {
    if (msg.role === "tool" && msg.tool_call_id !== void 0) {
      answered.add(String(msg.tool_call_id));
    }
  }
  const out = [];
  let cursor = 0;
  while (cursor < messages.length) {
    const msg = messages[cursor];
    out.push(msg);
    cursor += 1;
    if (msg.role !== "assistant" || !msg.tool_calls?.length) {
      continue;
    }
    while (cursor < messages.length && messages[cursor].role === "tool") {
      out.push(messages[cursor]);
      cursor += 1;
    }
    for (const call of msg.tool_calls) {
      const id = call.id ? String(call.id) : "";
      if (id && !answered.has(id)) {
        out.push({ role: "tool", tool_call_id: id, content: SYNTHETIC_TOOL_RESPONSE });
        answered.add(id);
      }
    }
  }
  messages.length = 0;
  messages.push(...out);
}
function applyGeminiFixups(messages, model) {
  if (!isGeminiModel(model)) {
    return;
  }
  mergeSystemMessages(messages);
  applyGeminiToolUseShim(messages);
  synthesizeMissingToolResponses(messages);
}

// src/translate/openai/image-parts.ts
function isImagePart(part) {
  return part.type === "input_image" || part.type === "image" || part.type === "image_url";
}
function imagePartToUrl(part) {
  const imgUrl = part.image_url;
  if (typeof imgUrl === "string") {
    return imgUrl;
  }
  if (imgUrl && typeof imgUrl === "object" && imgUrl.url) {
    return imgUrl.url;
  }
  const imgData = String(part.data ?? part.base64 ?? "");
  if (imgData) {
    const mimeType = String(part.mime_type ?? part.media_type ?? "image/png");
    return imgData.startsWith("data:") ? imgData : `data:${mimeType};base64,${imgData}`;
  }
  return "";
}

// src/translate/openai/thought-signature-tunnel.ts
var TUNNEL_SEP = "~gts~";
function encodeCallIdWithSignature(callId, signature) {
  if (!callId || !signature) {
    return callId;
  }
  if (callId.includes(TUNNEL_SEP) || signature.includes(TUNNEL_SEP)) {
    return callId;
  }
  return `${callId}${TUNNEL_SEP}${signature}`;
}
function decodeCallId(rawCallId) {
  const idx = rawCallId.indexOf(TUNNEL_SEP);
  if (idx === -1) {
    return { callId: rawCallId };
  }
  const signature = rawCallId.slice(idx + TUNNEL_SEP.length);
  return {
    callId: rawCallId.slice(0, idx),
    thoughtSignature: signature || void 0
  };
}

// src/translate/openai/translateRequest.ts
function translateRequest2(data, options = {}) {
  const messages = [];
  const systemContent = buildSystemContent(data.instructions);
  if (systemContent) {
    messages.push({ role: "system", content: systemContent });
  }
  const inputItems = typeof data.input === "string" ? [data.input] : Array.isArray(data.input) ? data.input : [];
  for (const raw of inputItems) {
    if (typeof raw === "string") {
      messages.push({ role: "user", content: raw });
      continue;
    }
    if (!raw || typeof raw !== "object") {
      continue;
    }
    const rawItem = raw;
    processInputItem(rawItem, messages, options);
  }
  const request = {
    model: data.model,
    messages
  };
  if (typeof data.temperature === "number") {
    request.temperature = data.temperature;
  }
  if (typeof data.top_p === "number") {
    request.top_p = data.top_p;
  }
  const effort = typeof data.reasoning?.effort === "string" ? data.reasoning.effort : void 0;
  if (effort) {
    const req = request;
    if (isGeminiModel(data.model)) {
      req.google = { thinking_config: { include_thoughts: true } };
    } else {
      req.reasoning_effort = effort;
    }
  }
  const maxTokens = typeof data.max_output_tokens === "number" && data.max_output_tokens || typeof data.max_tokens === "number" && data.max_tokens || options.defaultMaxTokens;
  if (typeof maxTokens === "number") {
    request.max_tokens = maxTokens;
  }
  const tools = mapTools2(data.tools ?? []);
  if (tools.length) {
    request.tools = tools;
    const toolChoice = mapToolChoice2(data.tool_choice);
    if (toolChoice !== void 0) {
      request.tool_choice = toolChoice;
    }
  }
  const placeholder = options.reasoningPlaceholder ?? ".";
  if (options.backfillReasoning !== false && placeholder) {
    for (const m of messages) {
      if (m.role === "assistant" && m.reasoning_content == null) {
        m.reasoning_content = placeholder;
      }
    }
  }
  repairToolMessageOrder(messages);
  applyGeminiFixups(messages, data.model);
  return { request };
}
function buildSystemContent(instructions) {
  if (!instructions) {
    return "";
  }
  if (typeof instructions === "string") {
    return instructions;
  }
  if (!Array.isArray(instructions)) {
    return "";
  }
  let out = "";
  for (const block of instructions) {
    if (typeof block === "string") {
      out += block;
    } else if (block && typeof block === "object") {
      out += String(block.text ?? "");
    }
  }
  return out;
}
function processInputItem(item, messages, options) {
  const itemType = String(item.type) || "message";
  const getLastAssistant = () => {
    const last = messages[messages.length - 1];
    if (last && last.role === "assistant") {
      return last;
    }
    const msg = { role: "assistant", content: null };
    messages.push(msg);
    return msg;
  };
  if (itemType === "message" || itemType === "agentMessage") {
    let role = String(item.role) || "user";
    if (role === "developer") {
      role = "system";
    }
    let reasoningContent = String(item.reasoning_content ?? "");
    const rawContent = item.content;
    if (role === "assistant" || role === "model") {
      let content = "";
      if (typeof rawContent === "string") {
        content = rawContent;
      } else if (Array.isArray(rawContent)) {
        for (const part of rawContent) {
          if (typeof part === "string") {
            content += part;
          } else if (part && typeof part === "object") {
            const contentPart = part;
            if (contentPart.type === "input_text" || contentPart.type === "text" || contentPart.type === "output_text") {
              content += String(contentPart.text ?? "");
            } else if (contentPart.type === "reasoning_text") {
              reasoningContent += String(contentPart.text ?? "");
            }
          }
        }
      }
      const amsg = getLastAssistant();
      if (content) {
        amsg.content = (amsg.content ?? "") + content;
      }
      if (reasoningContent) {
        amsg.reasoning_content = (amsg.reasoning_content ?? "") + reasoningContent;
      }
      const sig = item.thought_signature;
      if (typeof sig === "string" && sig) {
        amsg.thought_signature = sig;
      }
    } else {
      if (typeof rawContent === "string") {
        messages.push({ role, content: rawContent });
      } else if (Array.isArray(rawContent)) {
        const contentBlocks = [];
        for (const part of rawContent) {
          if (typeof part === "string") {
            contentBlocks.push({ type: "text", text: part });
          } else if (part && typeof part === "object") {
            const contentPart = part;
            if (contentPart.type === "input_text" || contentPart.type === "text" || contentPart.type === "output_text") {
              contentBlocks.push({ type: "text", text: String(contentPart.text ?? "") });
            } else if (contentPart.type === "reasoning_text") {
              reasoningContent += String(contentPart.text ?? "");
            } else if (isImagePart(contentPart)) {
              if (options.dropImages) {
                continue;
              }
              const url = imagePartToUrl(part);
              if (url) {
                contentBlocks.push({ type: "image_url", image_url: { url } });
              }
            } else if (part.type === "input_file" || part.type === "file") {
              const fileData = String(contentPart.file_data ?? contentPart.data ?? "");
              const mimeType = String(
                contentPart.mime_type ?? contentPart.media_type ?? "application/pdf"
              );
              if (fileData) {
                const url = fileData.startsWith("data:") ? fileData : `data:${mimeType};base64,${fileData}`;
                contentBlocks.push({ type: "image_url", image_url: { url } });
              }
            } else if (contentPart.type === "input_audio") {
              const ia = contentPart.input_audio;
              const data = String(ia?.data ?? contentPart.data ?? "");
              const format = String(ia?.format ?? contentPart.format ?? "mp3");
              if (data) {
                contentBlocks.push({ type: "input_audio", input_audio: { data, format } });
              }
            }
          }
        }
        const msg = { role, content: contentBlocks };
        if (reasoningContent) {
          msg.reasoning_content = reasoningContent;
        }
        const sig = item.thought_signature;
        if (typeof sig === "string" && sig) {
          msg.thought_signature = sig;
        }
        messages.push(msg);
      } else {
        messages.push({ role, content: "" });
      }
    }
    return;
  }
  if (itemType === "reasoning") {
    const rawList = item.content;
    let content = "";
    if (Array.isArray(rawList)) {
      for (const cp of rawList) {
        if (typeof cp === "string") {
          content += cp;
        } else if (cp && typeof cp === "object") {
          content += String(cp.text ?? "");
        }
      }
    } else if (typeof rawList === "string") {
      content += rawList;
    }
    const amsg = getLastAssistant();
    amsg.reasoning_content = (amsg.reasoning_content ?? "") + content;
    const sig = item.thought_signature;
    if (typeof sig === "string" && sig) {
      amsg.thought_signature = sig;
    }
    return;
  }
  if (itemType === "function_call" || itemType === "commandExecution" || itemType === "local_shell_call" || itemType === "fileChange" || itemType === "custom_tool_call" || itemType === "web_search_call") {
    processToolCall(item, messages, getLastAssistant, options);
    return;
  }
  if (itemType === "function_call_output" || itemType === "commandExecutionOutput" || itemType === "fileChangeOutput" || itemType === "custom_tool_call_output") {
    processToolOutput(item, messages, options);
    return;
  }
}
function processToolCall(item, messages, getLastAssistant, options) {
  const rawCallId = String(item.call_id ?? "") || String(item.id ?? "") || makeId("call");
  const decoded = options.tunnelThoughtSignatureInCallId ? decodeCallId(rawCallId) : { callId: rawCallId };
  const callId = decoded.callId;
  let name = item.name === void 0 ? void 0 : String(item.name);
  const itemType = item.type === void 0 ? void 0 : String(item.type);
  if (!name) {
    if (itemType === "commandExecution") {
      name = "run_shell_command";
    } else if (itemType === "local_shell_call") {
      name = "local_shell_command";
    } else if (itemType === "fileChange") {
      name = "write_file";
    } else if (itemType === "web_search_call") {
      name = "web_search";
    }
  }
  let args = item.arguments ?? item.input ?? (isEmpty(item.arguments) && isEmpty(item.input) ? {} : {});
  if (isEmpty(args) && itemType === "web_search_call") {
    args = item.action ?? {};
  }
  if (isEmpty(args)) {
    if (itemType === "commandExecution") {
      args = {
        command: item.command ?? "",
        dir_path: item.cwd ?? "."
      };
    } else if (itemType === "local_shell_call") {
      const action = item.action === void 0 ? {} : item.action;
      const execChild = action.exec === void 0 ? {} : action.exec;
      args = {
        command: execChild.command ?? [],
        working_directory: execChild.working_directory
      };
    } else if (itemType === "fileChange") {
      const changes = Array.isArray(item.changes) ? item.changes : [];
      const path = changes[0]?.path ?? "unknown";
      args = { file_path: path };
    }
  }
  const argsStr = typeof args === "string" ? args : jsonStringifySafe(args ?? {});
  if (!name) {
    return;
  }
  const amsg = getLastAssistant();
  if (!amsg.tool_calls) {
    amsg.tool_calls = [];
  }
  const toolCall = {
    id: callId,
    type: "function",
    function: { name, arguments: argsStr }
  };
  const sig = decoded.thoughtSignature ?? item.thought_signature ?? options.fallbackThoughtSignature;
  const thought = item.thought;
  if (typeof sig === "string" && sig) {
    toolCall.extra_content = { google: { thought_signature: sig } };
    amsg.thought_signature = sig;
  }
  amsg.tool_calls.push(toolCall);
  if (typeof thought === "string" && thought) {
    amsg.reasoning_content = (amsg.reasoning_content ?? "") + thought;
  }
}
function processToolOutput(item, messages, options) {
  const rawCallId = item.call_id === void 0 ? void 0 : String(item.call_id);
  const callId = rawCallId !== void 0 && options.tunnelThoughtSignatureInCallId ? decodeCallId(rawCallId).callId : rawCallId;
  const outputRaw = item.output ?? item.content ?? item.stdout ?? "";
  let content = "";
  const imageBlocks = [];
  if (typeof outputRaw === "string") {
    content = outputRaw;
  } else if (Array.isArray(outputRaw)) {
    for (const part of outputRaw) {
      if (typeof part === "string") {
        content += part;
      } else if (part && typeof part === "object") {
        const partItem = part;
        if (partItem.type === "input_text" || partItem.type === "text") {
          content += String(partItem.text ?? "");
        } else if (!options.dropImages && isImagePart(partItem)) {
          const url = imagePartToUrl(part);
          if (url) {
            imageBlocks.push({ type: "image_url", image_url: { url } });
          }
        }
      }
    }
  } else if (outputRaw && typeof outputRaw === "object") {
    const obj = outputRaw;
    content = String(obj.content ?? "");
    if (!content && obj.success === false) {
      content = "Error: Tool execution failed";
    }
  }
  if (!content && typeof item.stderr === "string" && item.stderr) {
    content = `Error: ${item.stderr}`;
  }
  messages.push({
    role: "tool",
    tool_call_id: callId,
    content
  });
  if (imageBlocks.length > 0) {
    messages.push({
      role: "user",
      content: [
        { type: "text", text: "[Image output returned by the preceding tool call]" },
        ...imageBlocks
      ]
    });
  }
}
function mapTools2(tools) {
  const out = [];
  for (const tool of tools) {
    if (!tool || typeof tool !== "object") {
      continue;
    }
    const tt = tool.type;
    if (tt === "function") {
      const fn = tool.function;
      const name = fn?.name ?? tool.name;
      if (!name) {
        continue;
      }
      const params = fn?.parameters ?? tool.parameters ?? { type: "object" };
      out.push({
        type: "function",
        function: {
          name,
          description: fn?.description ?? tool.description ?? "",
          parameters: params
        }
      });
      continue;
    }
    if (tt === "namespace") {
      const ns = tool.name;
      const nested = tool.tools;
      if (ns && Array.isArray(nested)) {
        for (const sub of nested) {
          if (!sub || typeof sub !== "object" || sub.type !== "function") {
            continue;
          }
          const subName = sub.name;
          if (!subName) {
            continue;
          }
          const params = sub.parameters ?? { type: "object" };
          out.push({
            type: "function",
            function: {
              name: `${ns}.${subName}`,
              description: sub.description ?? "",
              // eslint-disable-next-line no-restricted-syntax -- schema is Record<string,unknown> by OpenAPI convention
              parameters: params
            }
          });
        }
      }
      continue;
    }
  }
  return out;
}
function mapToolChoice2(choice) {
  if (choice == null) {
    return void 0;
  }
  if (choice === "auto" || choice === "required" || choice === "none") {
    return choice;
  }
  if (typeof choice === "object") {
    if (choice.type === "function" && "function" in choice && choice.function?.name) {
      return { type: "function", function: { name: choice.function.name } };
    }
    return choice;
  }
  return void 0;
}
function isEmpty(value) {
  if (value == null) {
    return true;
  }
  if (typeof value === "string") {
    return value.length === 0;
  }
  if (Array.isArray(value)) {
    return value.length === 0;
  }
  if (typeof value === "object" && value !== null) {
    return Object.keys(value).length === 0;
  }
  return false;
}
function repairToolMessageOrder(messages) {
  if (messages.length === 0) {
    return;
  }
  const toolById = /* @__PURE__ */ new Map();
  for (const msg of messages) {
    if (msg.role === "tool" && msg.tool_call_id !== void 0 && !toolById.has(String(msg.tool_call_id))) {
      toolById.set(String(msg.tool_call_id), msg);
    }
  }
  const used = /* @__PURE__ */ new Set();
  const out = [];
  for (const msg of messages) {
    if (msg.role === "tool") {
      continue;
    }
    if (msg.role !== "assistant") {
      out.push(msg);
      continue;
    }
    const calls = msg.tool_calls ?? [];
    if (calls.length === 0) {
      if (msg.content != null || msg.reasoning_content != null) {
        out.push(msg);
      }
      continue;
    }
    out.push(msg);
    for (const call of calls) {
      const id = call.id ? String(call.id) : "";
      const tool = id ? toolById.get(id) : void 0;
      if (id && tool && !used.has(id)) {
        out.push(tool);
        used.add(id);
      }
    }
  }
  messages.length = 0;
  messages.push(...out);
}

// src/translate/openai/translateResponse.ts
var SHELL_TOOL_NAMES3 = /* @__PURE__ */ new Set(["shell", "container.exec", "shell_command"]);
function buildShortNameToNamespace(tools) {
  const map = /* @__PURE__ */ new Map();
  for (const tool of tools) {
    if (!tool || typeof tool !== "object") {
      continue;
    }
    const entry = tool;
    const fn = entry.function;
    const flatName = typeof fn?.name === "string" ? fn.name : typeof entry.name === "string" ? entry.name : "";
    const dotIdx = flatName.indexOf(".");
    if (dotIdx !== -1) {
      map.set(flatName.slice(dotIdx + 1), flatName.slice(0, dotIdx));
      continue;
    }
    if (entry.type === "namespace" && typeof entry.name === "string" && Array.isArray(entry.tools)) {
      const ns = entry.name;
      for (const sub of entry.tools) {
        if (!sub || typeof sub !== "object") {
          continue;
        }
        const subEntry = sub;
        const subName = typeof subEntry.name === "string" ? subEntry.name : "";
        if (subName) {
          map.set(subName, ns);
        }
      }
    }
  }
  return map;
}
function translateResponse2(body, options = {}) {
  const createdAt = options.createdAt ?? body.created ?? Math.floor(Date.now() / 1e3);
  const id = options.responseId ?? body.id ?? makeId("resp");
  const model = options.model ?? body.model ?? "";
  const choice = body.choices?.[0];
  const message = choice?.message;
  const output = [];
  const shortNameToNs = buildShortNameToNamespace(options.requestTools ?? []);
  if (message?.tool_calls?.length) {
    for (const tc of message.tool_calls) {
      const item = mapToolCallToOutput(tc, shortNameToNs, options.tunnelThoughtSignatureInCallId);
      if (item) {
        output.push(item);
      }
    }
  }
  if (typeof message?.content === "string" && message.content) {
    output.push({
      id: makeId("msg"),
      type: "message",
      role: "assistant",
      status: "completed",
      content: [{ type: "output_text", text: message.content }]
    });
  }
  const usage = body.usage ?? {};
  const input = usage.prompt_tokens ?? 0;
  const completion = usage.completion_tokens ?? 0;
  const total = usage.total_tokens ?? input + completion;
  const cached = usage.prompt_tokens_details?.cached_tokens ?? 0;
  return {
    id,
    object: "response",
    created_at: createdAt,
    model,
    status: "completed",
    output,
    usage: {
      input_tokens: input,
      output_tokens: completion,
      total_tokens: total,
      input_tokens_details: {
        cached_tokens: cached
      }
    }
  };
}
function mapToolCallToOutput(tc, shortNameToNs, tunnelThoughtSignatureInCallId) {
  const name = tc.function?.name;
  if (!name) {
    return void 0;
  }
  const callId = tc.id ?? makeId("call");
  let args = tc.function?.arguments ?? "";
  if (typeof args !== "string") {
    args = jsonStringifySafe(args ?? {});
  }
  let resolvedName = name;
  let resolvedNamespace;
  if (!SHELL_TOOL_NAMES3.has(name)) {
    const dotIdx = name.indexOf(".");
    if (dotIdx !== -1) {
      resolvedNamespace = name.slice(0, dotIdx);
      resolvedName = name.slice(dotIdx + 1);
    } else {
      resolvedNamespace = shortNameToNs?.get(name);
    }
  }
  const item = {
    id: callId,
    type: "function_call",
    status: "completed",
    name: resolvedName,
    ...resolvedNamespace ? { namespace: resolvedNamespace } : {},
    arguments: args,
    call_id: callId
  };
  const thoughtSignature = getThoughtSignature(tc);
  if (thoughtSignature) {
    item.thought_signature = thoughtSignature;
    if (tunnelThoughtSignatureInCallId && item.call_id) {
      item.call_id = encodeCallIdWithSignature(item.call_id, thoughtSignature);
    }
  }
  if (SHELL_TOOL_NAMES3.has(resolvedName)) {
    item.type = "local_shell_call";
    const parsed = safeJsonParse(args);
    item.action = { type: "exec", command: parsed?.command ?? [] };
  }
  return item;
}
function getThoughtSignature(tc) {
  const sig = tc.extra_content?.google?.thought_signature ?? tc.thought_signature;
  return typeof sig === "string" && sig ? sig : void 0;
}

// src/translate/openai/translateStream.ts
var SHELL_TOOL_NAMES4 = /* @__PURE__ */ new Set(["shell", "container.exec", "shell_command"]);
function buildShortNameToNamespace2(tools) {
  const map = /* @__PURE__ */ new Map();
  for (const tool of tools) {
    if (!tool || typeof tool !== "object") {
      continue;
    }
    const entry = tool;
    const fn = entry.function;
    const flatName = typeof fn?.name === "string" ? fn.name : typeof entry.name === "string" ? entry.name : "";
    const dotIdx = flatName.indexOf(".");
    if (dotIdx !== -1) {
      map.set(flatName.slice(dotIdx + 1), flatName.slice(0, dotIdx));
      continue;
    }
    if (entry.type === "namespace" && typeof entry.name === "string" && Array.isArray(entry.tools)) {
      const ns = entry.name;
      for (const sub of entry.tools) {
        if (!sub || typeof sub !== "object") {
          continue;
        }
        const subEntry = sub;
        const subName = typeof subEntry.name === "string" ? subEntry.name : "";
        if (subName) {
          map.set(subName, ns);
        }
      }
    }
  }
  return map;
}
async function* translateStream2(stream, options = {}) {
  const translator = new StreamTranslator2(options);
  yield translator.createInitialEvent();
  for await (const msg of parseSseStream(stream)) {
    if (isDoneMessage(msg)) {
      break;
    }
    const chunk = safeJsonParse(msg.data);
    if (!chunk) {
      continue;
    }
    yield* translator.handleChunk(chunk);
  }
  yield* translator.finalize();
}
function isDoneMessage(msg) {
  return msg.data.trim() === "[DONE]";
}
function getToolCallKey(tc, ordinal) {
  if (typeof tc.index === "number") {
    return `index:${tc.index}`;
  }
  if (tc.id) {
    return `id:${tc.id}`;
  }
  return `ordinal:${ordinal}`;
}
var StreamTranslator2 = class {
  constructor(options) {
    __publicField(this, "model");
    __publicField(this, "responseId");
    __publicField(this, "createdAt");
    __publicField(this, "metadata");
    __publicField(this, "seq", 0);
    __publicField(this, "outputCounter", 0);
    __publicField(this, "textItem");
    __publicField(this, "textItemIndex", -1);
    __publicField(this, "textBuffer", "");
    // Reasoning (thinking) item — upstreams that expose chain-of-thought (DeepSeek,
    // Gemini OpenAI-compat, …) stream it on delta.reasoning_content. Surface it as a
    // Responses reasoning item so codex sees response.reasoning_text.delta.
    __publicField(this, "reasoningItem");
    __publicField(this, "reasoningItemIndex", -1);
    __publicField(this, "reasoningBuffer", "");
    __publicField(this, "toolCalls", /* @__PURE__ */ new Map());
    // shortName → namespace reverse map built from flattened namespace tools in
    // the translated request.  Used to restore the namespace when an upstream
    // (e.g. DeepSeek) omits the "namespace." prefix in its tool-call response.
    __publicField(this, "shortNameToNamespace");
    __publicField(this, "tunnelThoughtSignatureInCallId");
    __publicField(this, "inputTokens", 0);
    __publicField(this, "outputTokens", 0);
    __publicField(this, "cachedTokens", 0);
    this.model = options.model ?? "";
    this.responseId = options.responseId ?? makeId("resp");
    this.createdAt = options.createdAt ?? Math.floor(Date.now() / 1e3);
    this.metadata = options.requestMetadata ?? {};
    this.shortNameToNamespace = buildShortNameToNamespace2(this.metadata.tools ?? []);
    this.tunnelThoughtSignatureInCallId = options.tunnelThoughtSignatureInCallId ?? false;
  }
  createInitialEvent() {
    const toolsArr = this.metadata.tools ?? [];
    const response = {
      id: this.responseId,
      object: "response",
      created_at: this.createdAt,
      model: this.model,
      status: "in_progress",
      temperature: this.metadata.temperature,
      top_p: this.metadata.top_p,
      tool_choice: this.metadata.tool_choice,
      tools: toolsArr,
      parallel_tool_calls: true,
      store: this.metadata.store ?? true,
      metadata: this.metadata.metadata ?? {},
      output: []
    };
    return this.makeEvent("response.created", { response });
  }
  *handleChunk(chunk) {
    if (chunk.usage) {
      if (typeof chunk.usage.prompt_tokens === "number") {
        this.inputTokens = chunk.usage.prompt_tokens;
      }
      if (typeof chunk.usage.completion_tokens === "number") {
        this.outputTokens = chunk.usage.completion_tokens;
      }
      const cached = chunk.usage.prompt_tokens_details?.cached_tokens;
      if (typeof cached === "number") {
        this.cachedTokens = cached;
      }
    }
    const choice = chunk.choices?.[0];
    const delta = choice?.delta;
    if (!delta) {
      return;
    }
    if (delta.tool_calls?.length) {
      for (const [ordinal, tc] of delta.tool_calls.entries()) {
        const key = getToolCallKey(tc, ordinal);
        let state = this.toolCalls.get(key);
        if (!state) {
          const outputIndex = this.outputCounter++;
          const callId = tc.id ?? makeId("call");
          const item = {
            id: callId,
            type: "function_call",
            status: "in_progress",
            name: "",
            arguments: "",
            call_id: callId
          };
          state = { outputIndex, item };
          this.toolCalls.set(key, state);
          yield this.makeEvent("response.output_item.added", {
            response_id: this.responseId,
            output_index: outputIndex,
            item
          });
        }
        const fn = tc.function;
        const thoughtSignature = getThoughtSignature2(tc);
        if (thoughtSignature) {
          state.item.thought_signature = thoughtSignature;
        }
        if (fn?.name) {
          state.item.name = (state.item.name ?? "") + fn.name;
        }
        if (fn?.arguments != null) {
          const partial = typeof fn.arguments === "string" ? fn.arguments : jsonStringifySafe(fn.arguments);
          if (partial) {
            state.item.arguments = (state.item.arguments ?? "") + partial;
            yield this.makeEvent("response.function_call_arguments.delta", {
              response_id: this.responseId,
              item_id: state.item.id,
              output_index: state.outputIndex,
              delta: partial
            });
          }
        }
      }
    }
    const isThoughtContent = delta.extra_content?.google?.thought === true;
    const reasoningDelta = (typeof delta.reasoning_content === "string" ? delta.reasoning_content : "") + (isThoughtContent && typeof delta.content === "string" ? delta.content : "");
    const answerDelta = !isThoughtContent && typeof delta.content === "string" ? delta.content : "";
    if (reasoningDelta) {
      if (!this.reasoningItem) {
        const outputIndex = this.outputCounter++;
        this.reasoningItemIndex = outputIndex;
        this.reasoningItem = {
          id: makeId("rs"),
          type: "reasoning",
          summary: [],
          content: [{ type: "reasoning_text", text: "" }],
          status: "in_progress"
        };
        yield this.makeEvent("response.output_item.added", {
          response_id: this.responseId,
          output_index: outputIndex,
          item: this.reasoningItem
        });
      }
      this.reasoningBuffer += reasoningDelta;
      this.reasoningItem.content[0].text = this.reasoningBuffer;
      yield this.makeEvent("response.reasoning_text.delta", {
        response_id: this.responseId,
        item_id: this.reasoningItem.id,
        output_index: this.reasoningItemIndex,
        content_index: 0,
        delta: reasoningDelta
      });
    }
    if (answerDelta) {
      if (!this.textItem) {
        const outputIndex = this.outputCounter++;
        this.textItemIndex = outputIndex;
        this.textItem = {
          id: makeId("msg"),
          type: "message",
          role: "assistant",
          status: "in_progress",
          content: [{ type: "output_text", text: "" }]
        };
        yield this.makeEvent("response.output_item.added", {
          response_id: this.responseId,
          output_index: outputIndex,
          item: this.textItem
        });
      }
      this.textBuffer += answerDelta;
      const textContent = this.textItem.content[0];
      textContent.text = this.textBuffer;
      yield this.makeEvent("response.output_text.delta", {
        response_id: this.responseId,
        item_id: this.textItem.id,
        output_index: this.textItemIndex,
        content_index: 0,
        delta: answerDelta
      });
    }
  }
  *finalize() {
    const items = [];
    if (this.reasoningItem) {
      this.reasoningItem.status = "completed";
      items.push({ index: this.reasoningItemIndex, item: this.reasoningItem });
    }
    if (this.textItem) {
      this.textItem.status = "completed";
      items.push({ index: this.textItemIndex, item: this.textItem });
    }
    for (const state of this.toolCalls.values()) {
      const item = state.item;
      item.status = "completed";
      if (item.name && !SHELL_TOOL_NAMES4.has(item.name)) {
        const dotIdx = item.name.indexOf(".");
        if (dotIdx !== -1) {
          item.namespace = item.name.slice(0, dotIdx);
          item.name = item.name.slice(dotIdx + 1);
        } else {
          const ns = this.shortNameToNamespace.get(item.name);
          if (ns) {
            item.namespace = ns;
          }
        }
      }
      if (item.name && SHELL_TOOL_NAMES4.has(item.name)) {
        item.type = "local_shell_call";
        const parsed = safeJsonParse(item.arguments ?? "");
        item.action = { type: "exec", command: parsed?.command ?? [] };
      }
      if (this.tunnelThoughtSignatureInCallId && item.thought_signature && item.call_id) {
        item.call_id = encodeCallIdWithSignature(item.call_id, item.thought_signature);
      }
      items.push({ index: state.outputIndex, item });
    }
    items.sort((alpha, beta) => alpha.index - beta.index);
    for (const { index, item } of items) {
      yield this.makeEvent("response.output_item.done", {
        response_id: this.responseId,
        output_index: index,
        item
      });
    }
    const output = items.map((item) => item.item);
    const finalToolsArr = this.metadata.tools ?? [];
    const total = this.inputTokens + this.outputTokens;
    const response = {
      id: this.responseId,
      object: "response",
      created_at: this.createdAt,
      completed_at: Math.floor(Date.now() / 1e3),
      model: this.model,
      status: "completed",
      temperature: this.metadata.temperature,
      top_p: this.metadata.top_p,
      tool_choice: this.metadata.tool_choice,
      tools: finalToolsArr,
      parallel_tool_calls: true,
      store: this.metadata.store ?? true,
      metadata: this.metadata.metadata ?? {},
      output,
      usage: {
        input_tokens: this.inputTokens,
        output_tokens: this.outputTokens,
        total_tokens: total,
        input_tokens_details: {
          cached_tokens: this.cachedTokens
        }
      }
    };
    yield this.makeEvent("response.completed", { response });
  }
  makeEvent(type, data) {
    this.seq += 1;
    return {
      id: makeId("evt"),
      object: "response.event",
      type,
      created_at: Math.floor(Date.now() / 1e3),
      sequence_number: this.seq,
      ...data
    };
  }
};
function getThoughtSignature2(tc) {
  const sig = tc.extra_content?.google?.thought_signature ?? tc.thought_signature;
  return typeof sig === "string" && sig ? sig : void 0;
}

// src/tool-name-sanitizer.ts
function sanitizeUpstreamToolNames(body) {
  const map = /* @__PURE__ */ new Map();
  const tools = body.tools;
  if (!Array.isArray(tools)) {
    return map;
  }
  for (const tool of tools) {
    if (!tool || typeof tool !== "object") {
      continue;
    }
    const toolRecord = tool;
    const fn = toolRecord.function;
    if (!fn || typeof fn !== "object") {
      continue;
    }
    const fnRecord = fn;
    const original = fnRecord.name;
    if (typeof original !== "string") {
      continue;
    }
    const sanitized = original.replace(/[^a-zA-Z0-9_-]/g, "_");
    if (sanitized !== original) {
      map.set(sanitized, original);
      fnRecord.name = sanitized;
    }
  }
  return map;
}
function restoreToolNamesInChatResponse(body, map) {
  if (map.size === 0) {
    return body;
  }
  const choices = body.choices;
  if (!Array.isArray(choices)) {
    return body;
  }
  for (const choice of choices) {
    if (!choice || typeof choice !== "object") {
      continue;
    }
    const choiceRecord = choice;
    const message = choiceRecord.message;
    if (!message || typeof message !== "object") {
      continue;
    }
    const msg = message;
    const toolCalls = msg.tool_calls;
    if (!Array.isArray(toolCalls)) {
      continue;
    }
    for (const tc of toolCalls) {
      if (!tc || typeof tc !== "object") {
        continue;
      }
      const callRecord = tc;
      const fn = callRecord.function;
      if (!fn || typeof fn !== "object") {
        continue;
      }
      const fnRecord = fn;
      const name = fnRecord.name;
      if (typeof name === "string" && map.has(name)) {
        fnRecord.name = map.get(name);
      }
    }
  }
  return body;
}
function createToolNameRestoreStream(body, map) {
  if (map.size === 0) {
    return body;
  }
  const entries = Array.from(map.entries());
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  const reader = body.getReader();
  return new ReadableStream({
    async pull(controller) {
      try {
        const { value, done } = await reader.read();
        if (done) {
          controller.close();
          return;
        }
        let text = decoder.decode(value, { stream: true });
        for (const [sanitized, original] of entries) {
          const escapedSanitized = sanitized.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
          const re = new RegExp(`("name"\\s*:\\s*")${escapedSanitized}(")`, "g");
          text = text.replace(re, `$1${original}$2`);
        }
        controller.enqueue(encoder.encode(text));
      } catch (err) {
        controller.error(err);
      }
    },
    cancel() {
      reader.cancel().catch(() => {
      });
    }
  });
}

// src/utils/request.ts
function isResponsesEndpoint(url) {
  try {
    return /\/v1\/responses\/?$/.test(new URL(url, "http://_internal_").pathname);
  } catch {
    return /\/v1\/responses(?:\?|$)/.test(url);
  }
}
function urlOf(input) {
  if (typeof input === "string") {
    return input;
  }
  if (input instanceof URL) {
    return input.toString();
  }
  if (typeof Request !== "undefined" && input instanceof Request) {
    return input.url;
  }
  return String(input);
}
function parseHeaders(raw) {
  const out = {};
  if (!raw) {
    return out;
  }
  if (typeof Headers !== "undefined" && raw instanceof Headers) {
    raw.forEach((val, key) => {
      out[key.toLowerCase()] = val;
    });
    return out;
  }
  if (Array.isArray(raw)) {
    for (const [k, v] of raw) {
      out[String(k).toLowerCase()] = String(v);
    }
    return out;
  }
  for (const [k, v] of Object.entries(raw)) {
    out[k.toLowerCase()] = String(v);
  }
  return out;
}
function extractRequestMeta(input, init) {
  if (typeof Request !== "undefined" && input instanceof Request) {
    return { signal: input.signal, method: input.method, headers: parseHeaders(input.headers) };
  }
  return {
    signal: init?.signal ?? void 0,
    method: init?.method?.toUpperCase() ?? "GET",
    headers: parseHeaders(init?.headers)
  };
}
async function extractRequestBody(input, init) {
  if (typeof Request !== "undefined" && input instanceof Request) {
    const text = await input.text();
    return text || void 0;
  }
  return init?.body != null ? typeof init.body === "string" ? init.body : readBody(init.body) : void 0;
}
async function readBody(body) {
  if (typeof body === "string") {
    return body;
  }
  if (body instanceof Uint8Array || body instanceof ArrayBuffer) {
    return new TextDecoder().decode(body);
  }
  return String(body);
}

// src/fetch.ts
function createResponsesFetch(options) {
  if (!options.baseUrl) {
    throw new Error("baseUrl is required");
  }
  const rawFormat = options.upstreamFormat;
  const format = rawFormat ? normalizeFormat(rawFormat) : inferFormatFromUrl(options.baseUrl) ?? inferFormatFromModel(options.model) ?? "openai-chat";
  if (!format) {
    throw new Error(
      `Unsupported upstream format: ${options.upstreamFormat}. Use 'anthropic' or 'openai-chat'`
    );
  }
  const baseFetch = options.fetch ?? globalThis.fetch;
  if (!baseFetch) {
    throw new Error("fetch is not available; pass options.fetch");
  }
  const passthrough = options.passthroughFetch ?? baseFetch;
  return async (input, init) => {
    const url = urlOf(input);
    if (!isResponsesEndpoint(url)) {
      return passthrough(input, init);
    }
    const { signal, method, headers } = extractRequestMeta(input, init);
    if (method !== "POST") {
      return passthrough(input, init);
    }
    const body = await extractRequestBody(input, init);
    if (options.maxBodyChars !== void 0 && body !== void 0 && body.length > options.maxBodyChars) {
      return jsonErrorResponse(
        413,
        `Request body is ${(body.length / 1024 / 1024).toFixed(1)}MB, exceeding the configured ${(options.maxBodyChars / 1024 / 1024).toFixed(1)}MB limit. The conversation context (including inline media) has grown too large for this deployment \u2014 compact the conversation or avoid inlining large media.`,
        "invalid_request_error"
      );
    }
    let parsed;
    try {
      parsed = body ? JSON.parse(body) : void 0;
    } catch {
      return jsonErrorResponse(400, "Invalid JSON body for /responses");
    }
    if (!parsed) {
      return jsonErrorResponse(400, "Missing body for /responses");
    }
    if (options.model) {
      parsed.model = options.model;
    }
    return handleResponses(parsed, format, options, baseFetch, headers, signal, options.dropImages);
  };
}
function normalizeBaseUrl(url, format) {
  try {
    const parsedUrl = new URL(url);
    const path = parsedUrl.pathname.replace(/\/+$/, "");
    const endpoint = format === "anthropic" ? "/messages" : "/chat/completions";
    if (!path.endsWith(endpoint)) {
      parsedUrl.pathname = (path || "/v1") + endpoint;
    }
    return parsedUrl.toString();
  } catch {
    return url;
  }
}
function normalizeFormat(format) {
  const fmt = format === "anthropic" || format === "openai-chat" ? format : null;
  return fmt;
}
function inferFormatFromUrl(baseUrl) {
  try {
    const parsedUrl = new URL(baseUrl);
    const path = parsedUrl.pathname.replace(/\/+$/, "");
    if (/\/messages$/.test(path) || parsedUrl.hostname.toLowerCase().includes("anthropic")) {
      return "anthropic";
    }
    if (/\/chat\/completions$/.test(path)) {
      return "openai-chat";
    }
  } catch {
  }
  return null;
}
function inferFormatFromModel(model) {
  if (!model) {
    return null;
  }
  if (/^claude/i.test(model)) {
    return "anthropic";
  }
  return null;
}
async function handleResponses(request, format, options, baseFetch, incomingHeaders, signal, dropImages) {
  if (dropImages && options.fallbackUpstream && lastUserMessageHasImage(request)) {
    dropImages = false;
    const fb = options.fallbackUpstream;
    options = { ...options, ...fb, fallbackUpstream: void 0 };
    format = fb.upstreamFormat ?? inferFormatFromUrl(fb.baseUrl) ?? inferFormatFromModel(fb.model) ?? format;
    if (options.model) {
      request.model = options.model;
    }
    const fbModel = fb.model ? `, model: ${fb.model}` : "";
    console.warn(`[fallback] last user message has image, routing to ${fb.baseUrl}${fbModel}`);
  }
  if (options.dropTools && request.tools) {
    const { dropTools } = options;
    request = { ...request, tools: request.tools.filter((tool) => !dropTools(tool)) };
  }
  const streaming = request.stream ?? false;
  const resolvedUrl = normalizeBaseUrl(options.baseUrl, format);
  const { upstreamBody, requestMetadata } = buildUpstreamBody(
    request,
    format,
    streaming,
    options.baseUrl,
    dropImages,
    options.reasoning_effort,
    options.thinking,
    options.fallbackThoughtSignature,
    options.tunnelThoughtSignatureInCallId
  );
  const upstreamHeaders = buildUpstreamHeaders(format, options, incomingHeaders);
  const toolNameMap = format === "openai-chat" ? (
    // eslint-disable-next-line no-restricted-syntax -- upstreamBody is unknown; cast to Record for tool mutation
    sanitizeUpstreamToolNames(upstreamBody)
  ) : /* @__PURE__ */ new Map();
  const upstream = await baseFetch(resolvedUrl, {
    method: "POST",
    headers: upstreamHeaders,
    body: JSON.stringify(upstreamBody),
    signal
  });
  if (!upstream.ok) {
    return new Response(await upstream.text().catch(() => ""), {
      status: upstream.status,
      headers: { "content-type": "application/json" }
    });
  }
  if (!streaming) {
    const rawBody = await upstream.json();
    const restoredBody = restoreToolNamesInChatResponse(
      // eslint-disable-next-line no-restricted-syntax -- upstream json() returns unknown; cast to Record for tool name restoration
      rawBody,
      toolNameMap
    );
    const translated = format === "anthropic" ? (
      // eslint-disable-next-line no-restricted-syntax -- union type narrowing requires type assertion
      translateResponse(restoredBody, {
        model: request.model
      })
    ) : (
      // eslint-disable-next-line no-restricted-syntax -- union type narrowing requires type assertion
      translateResponse2(restoredBody, {
        model: request.model,
        requestTools: request.tools ?? [],
        tunnelThoughtSignatureInCallId: options.tunnelThoughtSignatureInCallId
      })
    );
    options.onCacheStats?.(extractCacheStatsFromResponse(translated));
    return new Response(JSON.stringify(translated), {
      status: 200,
      headers: { "content-type": "application/json" }
    });
  }
  if (!upstream.body) {
    return jsonErrorResponse(502, "Upstream streaming response has no body");
  }
  const upstreamBodyStream = createToolNameRestoreStream(upstream.body, toolNameMap);
  const events = format === "anthropic" ? translateStream(upstreamBodyStream, { model: request.model, requestMetadata }) : translateStream2(upstreamBodyStream, {
    model: request.model,
    requestMetadata,
    tunnelThoughtSignatureInCallId: options.tunnelThoughtSignatureInCallId
  });
  return new Response(
    responsesEventsToSseStream(collectCacheStatsFromStream(events, options.onCacheStats)),
    {
      status: 200,
      headers: {
        "content-type": "text/event-stream; charset=utf-8",
        "cache-control": "no-cache",
        connection: "keep-alive"
      }
    }
  );
}
function buildRequestMetadata(request, temperature, top_p) {
  return {
    temperature,
    top_p,
    tools: request.tools ?? [],
    tool_choice: request.tool_choice,
    store: request.store ?? true,
    metadata: request.metadata ?? {}
  };
}
function buildUpstreamBody(request, format, streaming, baseUrl, dropImages, reasoning_effort, thinking, fallbackThoughtSignature, tunnelThoughtSignatureInCallId) {
  if (format === "anthropic") {
    const { request: ar } = translateRequest(request);
    ar.stream = streaming;
    if (thinking !== void 0) {
      ar.thinking = thinking;
    } else if (reasoning_effort) {
      const effort = reasoning_effort.toLowerCase();
      if (isAdaptiveThinkingModel(ar.model)) {
        if (effort === "minimal") {
          delete ar.thinking;
          delete ar.output_config;
        } else {
          ar.thinking = { type: "adaptive" };
          const normalized = normalizeAnthropicEffort(effort);
          if (normalized) {
            ar.output_config = { ...ar.output_config ?? {}, effort: normalized };
          }
        }
      } else if (effort === "minimal") {
        ar.thinking = { type: "disabled" };
      } else if (effort === "low") {
        ar.thinking = { type: "enabled", budget_tokens: 4096 };
      } else if (effort === "medium") {
        ar.thinking = { type: "enabled", budget_tokens: 16384 };
      } else if (effort === "high") {
        ar.thinking = { type: "enabled", budget_tokens: 32768 };
      } else if (effort === "xhigh") {
        ar.thinking = { type: "enabled", budget_tokens: 65536 };
      }
    }
    if (ar.thinking && typeof ar.thinking === "object" && "budget_tokens" in ar.thinking) {
      const budget = Number(ar.thinking.budget_tokens);
      if (budget > 0 && ar.max_tokens <= budget) {
        ar.max_tokens = budget + 1024;
      }
    }
    return {
      upstreamBody: ar,
      requestMetadata: buildRequestMetadata(request, ar.temperature, ar.top_p)
    };
  }
  const { request: cr } = translateRequest2(request, {
    dropImages,
    fallbackThoughtSignature,
    tunnelThoughtSignatureInCallId
  });
  cr.stream = streaming;
  if (streaming) {
    cr.stream_options = { include_usage: true };
  }
  if (reasoning_effort !== void 0) {
    cr.reasoning_effort = reasoning_effort;
  }
  if (thinking !== void 0) {
    cr.thinking = thinking;
  }
  return {
    upstreamBody: cr,
    requestMetadata: buildRequestMetadata(request, cr.temperature, cr.top_p)
  };
}
function buildUpstreamHeaders(format, options, incoming) {
  const out = {};
  for (const [key, value] of Object.entries(incoming)) {
    if (DROPPED_HEADERS.has(key) || isClientSpecificHeader(key)) {
      continue;
    }
    out[key] = value;
  }
  if (options.defaultHeaders) {
    for (const [k, v] of Object.entries(options.defaultHeaders)) {
      out[k.toLowerCase()] = v;
    }
  }
  out["content-type"] = "application/json";
  if (format === "anthropic") {
    if (!out["anthropic-version"]) {
      out["anthropic-version"] = options.apiVersion ?? "2023-06-01";
    }
    if (typeof out["authorization"] === "string") {
      const match = /^Bearer\s+(.+)$/i.exec(out["authorization"]);
      if (match) {
        out["x-api-key"] = match[1].trim();
      }
    }
    delete out["authorization"];
  }
  return out;
}
var DROPPED_HEADERS = /* @__PURE__ */ new Set([
  "host",
  "content-length",
  "connection",
  "accept-encoding",
  "accept",
  "user-agent"
]);
function isClientSpecificHeader(key) {
  const keyLower = key.toLowerCase();
  return keyLower.startsWith("openai-") || keyLower.startsWith("x-stainless") || keyLower.startsWith("x-codex-") || keyLower === "originator" || keyLower === "session_id" || keyLower === "x-client-request-id";
}
function responsesEventsToSseStream(events) {
  const encoder = new TextEncoder();
  return new ReadableStream({
    async pull(controller) {
      try {
        const { value, done } = await events.next();
        if (done) {
          controller.enqueue(encoder.encode("data: [DONE]\n\n"));
          controller.close();
          return;
        }
        controller.enqueue(encoder.encode(encodeSseEvent(value && value.type, value)));
      } catch (err) {
        controller.error(err);
      }
    },
    async cancel() {
      try {
        await events.return?.();
      } catch {
      }
    }
  });
}
function extractCacheStatsFromResponse(response) {
  const usage = response.usage;
  return {
    cachedTokens: usage?.input_tokens_details?.cached_tokens ?? 0,
    cacheCreationTokens: usage?.input_tokens_details?.cache_creation_tokens ?? 0,
    inputTokens: usage?.input_tokens ?? 0,
    outputTokens: usage?.output_tokens ?? 0,
    totalTokens: usage?.total_tokens ?? 0
  };
}
async function* collectCacheStatsFromStream(events, onCacheStats) {
  let lastStats;
  for await (const event of events) {
    if (event.type === "response.completed") {
      const eventResp = event;
      const resp = eventResp.response;
      if (resp?.usage) {
        lastStats = extractCacheStatsFromResponse(resp);
      }
    }
    yield event;
  }
  if (lastStats && onCacheStats) {
    onCacheStats(lastStats);
  }
}
function lastUserMessageHasImage(request) {
  const input = request.input;
  if (!input || !Array.isArray(input)) {
    return false;
  }
  for (let i = input.length - 1; i >= 0; i--) {
    const item = input[i];
    if (!item || typeof item !== "object") {
      continue;
    }
    const itemRecord = item;
    if (itemRecord.role !== "user") {
      continue;
    }
    const content = itemRecord.content;
    if (!Array.isArray(content)) {
      return false;
    }
    for (const part of content) {
      if (part && typeof part === "object") {
        const partItem = part;
        const type = partItem.type;
        if (type === "input_image" || type === "image" || type === "image_url") {
          return true;
        }
      }
    }
    return false;
  }
  return false;
}
function jsonErrorResponse(status, message, type = "upstream_error") {
  return new Response(JSON.stringify({ error: { message, type, code: String(status) } }), {
    status,
    headers: { "content-type": "application/json" }
  });
}

// src/translate/index.ts
var translate_exports = {};
__export(translate_exports, {
  anthropic: () => anthropic_exports,
  openai: () => openai_exports
});

exports.createResponsesFetch = createResponsesFetch;
exports.encodeSseEvent = encodeSseEvent;
exports.parseSseStream = parseSseStream;
exports.translate = translate_exports;
//# sourceMappingURL=index.cjs.map
//# sourceMappingURL=index.cjs.map