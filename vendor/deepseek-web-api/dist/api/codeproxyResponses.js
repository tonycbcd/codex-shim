import { createResponsesFetch, parseSseStream } from "../../../codeproxy-core/dist/index.js";

const encoder = new TextEncoder();

function chatChunkSse(chunk) {
    return encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`);
}

function doneSse() {
    return encoder.encode("data: [DONE]\n\n");
}

/**
 * Convert a Codex Responses request with the vendored @codeproxy/core adapter,
 * run it through DeepSeek Web's Chat Completions implementation, then feed the
 * resulting standard Chat chunks back through codeproxy's stream translator.
 *
 * The browser-specific XML/JSON recovery stays inside DeepSeekClient.streamChat.
 * This module owns the protocol conversion so it cannot drift from the mature
 * codeproxy implementation.
 */
function nativeToolName(tool) {
    if (typeof tool?.name === "string" && tool.name) {
        return tool.name;
    }
    const type = String(tool?.type ?? "");
    if (type === "apply_patch" || type === "custom") {
        return "apply_patch";
    }
    if (type === "local_shell" || type === "shell") {
        return "local_shell";
    }
    if (type.startsWith("web_search")) {
        return "web_search";
    }
    return type;
}

function nativeToolParameters(tool) {
    if (tool?.parameters && typeof tool.parameters === "object") {
        return tool.parameters;
    }
    const type = String(tool?.type ?? "");
    if (type === "apply_patch" || type === "custom") {
        return {
            type: "object",
            properties: {
                patch: {
                    type: "string",
                    description: "Codex patch from *** Begin Patch through *** End Patch.",
                },
            },
            required: ["patch"],
            additionalProperties: true,
        };
    }
    if (type === "local_shell" || type === "shell") {
        return {
            type: "object",
            properties: {
                command: { type: "array", items: { type: "string" } },
            },
            required: ["command"],
            additionalProperties: true,
        };
    }
    if (type.startsWith("web_search")) {
        return {
            type: "object",
            properties: { query: { type: "string" } },
            required: ["query"],
            additionalProperties: true,
        };
    }
    return { type: "object", additionalProperties: true };
}

function normalizeNativeTools(tools) {
    const nativeTypes = new Map();
    const normalized = [];
    for (const tool of Array.isArray(tools) ? tools : []) {
        if (!tool || typeof tool !== "object") {
            continue;
        }
        if (tool.type === "function" || tool.type === "namespace") {
            normalized.push(tool);
            continue;
        }
        const name = nativeToolName(tool);
        if (!name) {
            continue;
        }
        nativeTypes.set(name, String(tool.type ?? ""));
        normalized.push({
            type: "function",
            name,
            description: tool.description ?? `Codex native tool ${tool.type}`,
            parameters: nativeToolParameters(tool),
        });
    }
    return { tools: normalized, nativeTypes };
}

function normalizeCodexRequest(body) {
    const normalized = structuredClone(body);
    if (Array.isArray(normalized.input)) {
        normalized.input = normalized.input.map((item) => {
            if (item && typeof item === "object" && !item.type && item.role) {
                return { ...item, type: "message" };
            }
            return item;
        });
    }
    const native = normalizeNativeTools(normalized.tools);
    normalized.tools = native.tools;
    // Always use the streaming upstream path. It preserves reasoning and
    // fragmented/multiple tool calls; non-stream callers are accumulated from
    // the canonical response.completed event below.
    normalized.stream = true;
    return { request: normalized, nativeTypes: native.nativeTypes };
}

function customToolInput(argumentsText) {
    try {
        const parsed = JSON.parse(argumentsText || "{}");
        if (typeof parsed === "string") {
            return parsed;
        }
        if (parsed && typeof parsed === "object") {
            if (typeof parsed.patch === "string") {
                return parsed.patch;
            }
            if (typeof parsed.input === "string") {
                return parsed.input;
            }
        }
    }
    catch {
        // A free-form custom tool may already contain its raw input.
    }
    return argumentsText ?? "";
}

function nativeEventAdapter(nativeTypes) {
    const customIndexes = new Set();
    function adaptItem(item) {
        if (
            item?.type !== "function_call" ||
            !["apply_patch", "custom"].includes(nativeTypes.get(item.name))
        ) {
            return item;
        }
        return {
            ...item,
            type: "custom_tool_call",
            input: customToolInput(item.arguments),
            arguments: undefined,
        };
    }
    return function adapt(event) {
        const item = event?.item;
        if (
            item?.type === "function_call" &&
            nativeTypes.get(item.name) &&
            (nativeTypes.get(item.name) === "apply_patch" ||
                nativeTypes.get(item.name) === "custom")
        ) {
            customIndexes.add(event.output_index);
            const adaptedItem = adaptItem(item);
            if (event.type === "response.output_item.done") {
                return [
                    {
                        type: "response.custom_tool_call_input.delta",
                        item_id: adaptedItem.id,
                        output_index: event.output_index,
                        delta: adaptedItem.input,
                    },
                    {
                        type: "response.custom_tool_call_input.done",
                        item_id: adaptedItem.id,
                        output_index: event.output_index,
                        input: adaptedItem.input,
                    },
                    { ...event, item: adaptedItem },
                ];
            }
            return [{
                ...event,
                item: adaptedItem,
            }];
        }
        if (
            event?.type === "response.function_call_arguments.delta" &&
            customIndexes.has(event.output_index)
        ) {
            // JSON argument fragments are not valid free-form custom-tool
            // input. The complete normalized patch is emitted when the item
            // closes.
            return [];
        }
        if (
            event?.type === "response.function_call_arguments.done" &&
            customIndexes.has(event.output_index)
        ) {
            return [{
                ...event,
                type: "response.custom_tool_call_input.done",
                input: customToolInput(event.arguments),
                arguments: undefined,
            }];
        }
        if (event?.type === "response.completed" && Array.isArray(event.response?.output)) {
            return [{
                ...event,
                response: {
                    ...event.response,
                    output: event.response.output.map(adaptItem),
                },
            }];
        }
        return [event];
    };
}

function deepSeekSourceTag(model) {
    return String(model ?? "").toLowerCase().includes("pro")
        ? "[deepseek-pro]"
        : "[deepseek]";
}

function sourceEventAdapter(model) {
    const tag = deepSeekSourceTag(model);
    let firstTextDelta = true;

    function prefixText(text) {
        if (typeof text !== "string" || !text) {
            return text;
        }
        if (/^\[deepseek(?:-pro)?\]\s*/i.test(text)) {
            return text;
        }
        return `${tag} ${text}`;
    }

    function adaptMessageItem(item) {
        if (item?.type !== "message" || !Array.isArray(item.content)) {
            return item;
        }
        let prefixed = false;
        return {
            ...item,
            content: item.content.map((part) => {
                if (
                    !prefixed &&
                    part?.type === "output_text" &&
                    typeof part.text === "string" &&
                    part.text
                ) {
                    prefixed = true;
                    return { ...part, text: prefixText(part.text) };
                }
                return part;
            }),
        };
    }

    return function adapt(event) {
        if (
            event?.type === "response.output_text.delta" &&
            firstTextDelta &&
            typeof event.delta === "string" &&
            event.delta
        ) {
            firstTextDelta = false;
            return { ...event, delta: prefixText(event.delta) };
        }
        if (
            event?.type === "response.output_text.done" &&
            typeof event.text === "string"
        ) {
            return { ...event, text: prefixText(event.text) };
        }
        if (
            event?.type === "response.output_item.done" &&
            event.item?.type === "message"
        ) {
            return { ...event, item: adaptMessageItem(event.item) };
        }
        if (
            event?.type === "response.completed" &&
            Array.isArray(event.response?.output)
        ) {
            return {
                ...event,
                response: {
                    ...event.response,
                    output: event.response.output.map(adaptMessageItem),
                },
            };
        }
        return event;
    };
}

function deepSeekChatFetch(client) {
    return async (_input, init = {}) => {
        const chatRequest = JSON.parse(String(init.body ?? "{}"));
        chatRequest.stream = true;
        chatRequest.stream_options = {
            ...(chatRequest.stream_options ?? {}),
            include_usage: true,
        };
        const transform = new TransformStream();
        const writer = transform.writable.getWriter();
        let writes = Promise.resolve();

        const producer = (async () => {
            try {
                await client.streamChat(chatRequest, (chunk) => {
                    writes = writes.then(() => writer.write(chatChunkSse(chunk)));
                });
                await writes;
                await writer.write(doneSse());
                await writer.close();
            }
            catch (error) {
                await writes.catch(() => undefined);
                await writer.abort(error).catch(() => undefined);
                throw error;
            }
        })();
        // The stream communicates producer failures to codeproxy. Also attach a
        // rejection handler so an early downstream disconnect is not reported
        // as an unhandled promise rejection.
        void producer.catch(() => undefined);
        return new Response(transform.readable, {
            status: 200,
            headers: { "content-type": "text/event-stream; charset=utf-8" },
        });
    };
}

export async function* streamCodeproxyResponses(body, client) {
    const normalized = normalizeCodexRequest(body);
    const requestBody = normalized.request;
    const adaptNativeEvent = nativeEventAdapter(normalized.nativeTypes);
    const adaptSourceEvent = sourceEventAdapter(requestBody.model);
    const responsesFetch = createResponsesFetch({
        baseUrl: "https://deepseek-web.invalid/v1",
        upstreamFormat: "openai-chat",
        model: requestBody.model,
        dropImages: true,
        fetch: deepSeekChatFetch(client),
        timeoutMs: 120_000,
    });
    const response = await responsesFetch("https://codex-shim.invalid/v1/responses", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(requestBody),
    });
    if (!response.ok || !response.body) {
        throw new Error(`codeproxy translation failed (${response.status}): ${await response.text()}`);
    }
    for await (const message of parseSseStream(response.body)) {
        if (!message.data || message.data === "[DONE]") {
            continue;
        }
        const event = JSON.parse(message.data);
        if (event && typeof event === "object") {
            for (const adapted of adaptNativeEvent(event)) {
                yield adaptSourceEvent(adapted);
            }
        }
    }
}

/** Accumulate the canonical response.completed payload for non-stream calls. */
export async function completeCodeproxyResponses(body, client) {
    let completed;
    for await (const event of streamCodeproxyResponses(body, client)) {
        if (event?.type === "response.completed" && event.response) {
            completed = event.response;
        }
    }
    if (!completed) {
        throw new Error("DeepSeek Web stream ended without response.completed");
    }
    return completed;
}
