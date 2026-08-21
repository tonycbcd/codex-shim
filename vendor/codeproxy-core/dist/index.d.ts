/**
 * OpenAI Responses API type definitions (subset used by this library).
 * Only the fields relevant to request translation and event emission are typed.
 * Other fields pass through as `unknown` / `Record<string, unknown>`.
 */
type ResponsesInputItem = string | ResponsesMessageItem | ResponsesReasoningItem | ResponsesFunctionCallItem | ResponsesFunctionCallOutputItem | ResponsesLocalShellCallItem | ResponsesCommandExecutionItem | ResponsesCommandExecutionOutputItem | ResponsesCustomToolCallItem | ResponsesCustomToolCallOutputItem | ResponsesFileChangeItem | ResponsesFileChangeOutputItem | ResponsesWebSearchCallItem | Record<string, unknown>;
interface ResponsesContentPart {
    type: 'input_text' | 'text' | 'output_text' | 'reasoning_text' | 'input_image' | 'image' | 'image_url' | 'input_file' | 'file' | 'input_audio' | 'tool_result' | string;
    text?: string;
    image_url?: string | {
        url: string;
    };
    source?: Record<string, unknown>;
    data?: string;
    base64?: string;
    media_type?: string;
    mime_type?: string;
    file_data?: string;
    file_url?: string | {
        url: string;
    };
    input_audio?: {
        data?: string;
        format?: string;
    };
    format?: string;
    tool_use_id?: string;
    call_id?: string;
    content?: unknown;
    cache_control?: Record<string, unknown>;
    [key: string]: unknown;
}
interface ResponsesMessageItem {
    type?: 'message' | 'agentMessage';
    role?: 'user' | 'assistant' | 'model' | 'system' | 'developer' | string;
    content?: string | ResponsesContentPart[] | unknown;
    reasoning_content?: string;
    thought_signature?: string;
    [key: string]: unknown;
}
interface ResponsesReasoningItem {
    type: 'reasoning';
    content?: Array<{
        type?: string;
        text?: string;
    } | string> | string;
    thought_signature?: string;
    [key: string]: unknown;
}
interface ResponsesFunctionCallItem {
    type: 'function_call';
    id?: string;
    call_id?: string;
    name?: string;
    arguments?: string | Record<string, unknown>;
    thought_signature?: string;
    [key: string]: unknown;
}
interface ResponsesFunctionCallOutputItem {
    type: 'function_call_output';
    id?: string;
    call_id?: string;
    output?: unknown;
    [key: string]: unknown;
}
interface ResponsesLocalShellCallItem {
    type: 'local_shell_call';
    id?: string;
    call_id?: string;
    action?: {
        type?: string;
        exec?: {
            command?: string[];
            working_directory?: string;
        };
        command?: string[];
    };
    [key: string]: unknown;
}
interface ResponsesCommandExecutionItem {
    type: 'commandExecution';
    id?: string;
    call_id?: string;
    command?: string;
    cwd?: string;
    [key: string]: unknown;
}
interface ResponsesCommandExecutionOutputItem {
    type: 'commandExecutionOutput';
    id?: string;
    call_id?: string;
    output?: unknown;
    stdout?: string;
    stderr?: string;
    [key: string]: unknown;
}
interface ResponsesCustomToolCallItem {
    type: 'custom_tool_call';
    id?: string;
    call_id?: string;
    name?: string;
    input?: unknown;
    [key: string]: unknown;
}
interface ResponsesCustomToolCallOutputItem {
    type: 'custom_tool_call_output';
    id?: string;
    call_id?: string;
    output?: unknown;
    [key: string]: unknown;
}
interface ResponsesFileChangeItem {
    type: 'fileChange';
    id?: string;
    call_id?: string;
    changes?: Array<{
        path?: string;
        [k: string]: unknown;
    }>;
    [key: string]: unknown;
}
interface ResponsesFileChangeOutputItem {
    type: 'fileChangeOutput';
    id?: string;
    call_id?: string;
    output?: unknown;
    [key: string]: unknown;
}
interface ResponsesWebSearchCallItem {
    type: 'web_search_call';
    id?: string;
    call_id?: string;
    action?: Record<string, unknown>;
    [key: string]: unknown;
}
interface ResponsesTool {
    type: 'function' | string;
    name?: string;
    description?: string;
    parameters?: Record<string, unknown>;
    strict?: boolean;
    function?: {
        name: string;
        description?: string;
        parameters?: Record<string, unknown>;
        strict?: boolean;
    };
    [key: string]: unknown;
}
type ResponsesToolChoice = 'auto' | 'required' | 'none' | {
    type: 'function';
    function: {
        name: string;
    };
} | {
    type: 'auto' | 'any';
} | null | undefined;
interface ResponsesReasoningConfig {
    effort?: 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | string;
    summary?: string;
    [key: string]: unknown;
}
interface ResponsesRequest {
    model: string;
    input?: ResponsesInputItem[] | string;
    instructions?: string | Array<string | {
        text?: string;
    }>;
    tools?: ResponsesTool[];
    tool_choice?: ResponsesToolChoice;
    temperature?: number;
    top_p?: number;
    max_output_tokens?: number;
    max_tokens?: number;
    stream?: boolean;
    reasoning?: ResponsesReasoningConfig;
    metadata?: Record<string, unknown>;
    store?: boolean;
    previous_response_id?: string | null;
    [key: string]: unknown;
}
interface ResponsesOutputMessage {
    id: string;
    type: 'message';
    role: 'assistant';
    status: 'completed' | 'in_progress';
    content: Array<{
        type: 'output_text';
        text: string;
    } | {
        type: string;
        [k: string]: unknown;
    }>;
}
interface ResponsesOutputFunctionCall {
    id: string;
    type: 'function_call' | 'local_shell_call' | string;
    status: 'completed' | 'in_progress';
    name?: string;
    namespace?: string;
    arguments?: string;
    call_id?: string;
    thought_signature?: string;
    action?: {
        type?: string;
        command?: string[];
    };
}
interface ResponsesOutputReasoning {
    id: string;
    type: 'reasoning';
    summary: unknown[];
    content: Array<{
        type: 'reasoning_text';
        text: string;
    }>;
    status?: 'completed' | 'in_progress';
}
type ResponsesOutputItem = ResponsesOutputMessage | ResponsesOutputFunctionCall | ResponsesOutputReasoning | Record<string, unknown>;
interface ResponsesUsage {
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
    input_tokens_details?: {
        cached_tokens?: number;
        cache_creation_tokens?: number;
    };
}
interface ResponsesResponse {
    id: string;
    object: 'response';
    created_at: number;
    completed_at?: number;
    model: string;
    status: 'completed' | 'in_progress' | 'failed';
    output: ResponsesOutputItem[];
    usage?: ResponsesUsage;
    temperature?: number;
    top_p?: number;
    tool_choice?: unknown;
    tools?: unknown[];
    parallel_tool_calls?: boolean;
    store?: boolean;
    metadata?: Record<string, unknown>;
    [key: string]: unknown;
}
interface ResponsesStreamEvent {
    id: string;
    object: 'response.event';
    type: string;
    created_at: number;
    sequence_number: number;
    [key: string]: unknown;
}

type UpstreamFormat = 'anthropic' | 'openai-chat';
interface CacheStats {
    cachedTokens: number;
    cacheCreationTokens: number;
    inputTokens: number;
    outputTokens: number;
    totalTokens: number;
}
interface CreateResponsesFetchOptions {
    /** Upstream API format. If omitted, inferred from `baseUrl`. */
    upstreamFormat?: UpstreamFormat;
    /** Upstream endpoint URL. Required. */
    baseUrl: string;
    /** Override upstream API version header (Anthropic only). */
    apiVersion?: string;
    /** Replace the caller-provided `model` field before translation. */
    model?: string;
    /** Extra headers merged into every upstream call. */
    defaultHeaders?: Record<string, string>;
    /** Underlying fetch. Defaults to `globalThis.fetch`. */
    fetch?: typeof fetch;
    /** Non-/responses traffic forward target. Defaults to `options.fetch`. */
    passthroughFetch?: typeof fetch;
    /** Drop image/file parts from user messages (e.g. DeepSeek text-only models). */
    dropImages?: boolean;
    /** Drop tools from the request before translation. Return true to drop a tool.
     *  Use to reduce payload size when the upstream has a limit (e.g. DeepSeek ECONNRESET
     *  on 252 KB requests with 143 tools). Applied before namespace flattening. */
    dropTools?: (tool: ResponsesTool) => boolean;
    /** Reject /responses bodies longer than this many characters (≈ bytes for the
     *  ASCII-dominant JSON these APIs exchange) with a 413 error envelope BEFORE
     *  parse/translate. The translation pipeline (parse → translate → re-serialize)
     *  holds several times the body size at peak, so on memory-capped runtimes
     *  (e.g. 128 MB Cloudflare Workers isolates) one runaway request can OOM the
     *  whole isolate and kill every concurrent request. Off by default. */
    maxBodyChars?: number;
    /** Fallback thought signature for Gemini OpenAI-compatible tool histories. */
    fallbackThoughtSignature?: string;
    /** Tunnel the Gemini thought signature through the function_call `call_id` so
     *  it survives clients that drop the dedicated `thought_signature` field (e.g.
     *  codex-ts, whose protocol mirrors codex-rs). Encoded on the response and
     *  stripped back off on the next request. Off by default. */
    tunnelThoughtSignatureInCallId?: boolean;
    /** Optional callback to receive cache statistics. */
    onCacheStats?: (stats: CacheStats) => void;
    /** Override reasoning_effort sent to the upstream model (OpenAI Chat / Anthropic). */
    reasoning_effort?: string;
    /** Override thinking configuration sent to the upstream model. */
    thinking?: unknown;
    /** Timeout in milliseconds for upstream requests. Defaults to no timeout. */
    timeoutMs?: number;
    /** Fallback upstream for requests where the last user message contains images. */
    fallbackUpstream?: {
        baseUrl: string;
        upstreamFormat?: UpstreamFormat;
        model?: string;
        defaultHeaders?: Record<string, string>;
        apiVersion?: string;
        reasoning_effort?: string;
        thinking?: unknown;
    };
}
declare function createResponsesFetch(options: CreateResponsesFetchOptions): typeof fetch;

/** Anthropic Messages API type definitions (subset). */
interface AnthropicTextBlock {
    type: 'text';
    text: string;
    cache_control?: Record<string, unknown>;
}
interface AnthropicImageSource {
    type: 'base64' | 'url';
    media_type?: string;
    data?: string;
    url?: string;
}
interface AnthropicImageBlock {
    type: 'image';
    source: AnthropicImageSource;
    cache_control?: Record<string, unknown>;
}
interface AnthropicDocumentBlock {
    type: 'document';
    source: AnthropicImageSource;
    cache_control?: Record<string, unknown>;
}
interface AnthropicToolUseBlock {
    type: 'tool_use';
    id: string;
    name: string;
    input: Record<string, unknown>;
    cache_control?: Record<string, unknown>;
}
interface AnthropicToolResultBlock {
    type: 'tool_result';
    tool_use_id: string;
    content: string | AnthropicContentBlock[];
    is_error?: boolean;
    cache_control?: Record<string, unknown>;
}
interface AnthropicThinkingBlock {
    type: 'thinking';
    thinking: string;
    signature?: string;
}
type AnthropicContentBlock = AnthropicTextBlock | AnthropicImageBlock | AnthropicDocumentBlock | AnthropicToolUseBlock | AnthropicToolResultBlock | AnthropicThinkingBlock | {
    type: string;
    [k: string]: unknown;
};
interface AnthropicMessage {
    role: 'user' | 'assistant';
    content: AnthropicContentBlock[] | string;
}
interface AnthropicTool {
    name: string;
    description?: string;
    input_schema: Record<string, unknown>;
}
type AnthropicToolChoice = {
    type: 'auto';
} | {
    type: 'any';
} | {
    type: 'tool';
    name: string;
};
interface AnthropicThinkingConfig {
    type: 'enabled' | 'disabled' | 'adaptive';
    budget_tokens?: number;
}
/** Anthropic `output_config` (subset). On adaptive-generation models, thinking
 *  depth is controlled by `effort` here — NOT by a token budget on `thinking`. */
interface AnthropicOutputConfig {
    effort?: 'low' | 'medium' | 'high' | 'xhigh' | 'max' | string;
}
interface AnthropicRequest {
    model: string;
    messages: AnthropicMessage[];
    system?: string | AnthropicTextBlock[];
    max_tokens: number;
    temperature?: number;
    top_p?: number;
    tools?: (AnthropicTool | Record<string, unknown>)[];
    tool_choice?: AnthropicToolChoice;
    metadata?: Record<string, unknown>;
    thinking?: AnthropicThinkingConfig;
    output_config?: AnthropicOutputConfig;
    stream?: boolean;
    [k: string]: unknown;
}
interface AnthropicUsage {
    input_tokens: number;
    output_tokens: number;
    cache_creation_input_tokens?: number;
    cache_read_input_tokens?: number;
}
interface AnthropicResponse {
    id: string;
    type: 'message';
    role: 'assistant';
    model: string;
    content: AnthropicContentBlock[];
    stop_reason?: string;
    stop_sequence?: string | null;
    usage: AnthropicUsage;
}
/** Anthropic streaming SSE events. */
type AnthropicStreamEvent = {
    type: 'message_start';
    message: AnthropicResponse;
} | {
    type: 'content_block_start';
    index: number;
    content_block: AnthropicContentBlock;
} | {
    type: 'content_block_delta';
    index: number;
    delta: {
        type: 'text_delta';
        text: string;
    } | {
        type: 'thinking_delta';
        thinking: string;
    } | {
        type: 'input_json_delta';
        partial_json: string;
    } | {
        type: 'signature_delta';
        signature: string;
    } | {
        type: string;
        [k: string]: unknown;
    };
} | {
    type: 'content_block_stop';
    index: number;
} | {
    type: 'message_delta';
    delta: Record<string, unknown>;
    usage?: AnthropicUsage;
} | {
    type: 'message_stop';
} | {
    type: 'ping';
} | {
    type: string;
    [k: string]: unknown;
};

interface TranslateRequestOptions$1 {
    /** Default max tokens when not provided (Anthropic requires `max_tokens`). */
    defaultMaxTokens?: number;
    /** Thinking budget overrides, keyed by effort. */
    reasoningBudgets?: Partial<Record<'minimal' | 'low' | 'medium' | 'high' | 'xhigh', number>>;
}
interface TranslateRequestResult$1 {
    request: AnthropicRequest;
    hasPromptCache: boolean;
}
/** Convert a Responses API request into an Anthropic Messages API request. */
declare function translateRequest$1(data: ResponsesRequest, options?: TranslateRequestOptions$1): TranslateRequestResult$1;
/**
 * Claude "adaptive-generation" models — Sonnet 5+, Opus 4.6+, and the
 * Fable/Mythos 5 family — use `thinking: {type:'adaptive'}` and REJECT the
 * legacy `{type:'enabled', budget_tokens}` with a 400. Older Claude models
 * (Opus ≤4.5, Sonnet ≤4.5, every Haiku shipped so far) and non-Claude models
 * keep the budget_tokens shape. On adaptive models, thinking depth is set by
 * `output_config.effort`, not a token budget.
 *
 * Tolerant of provider prefixes (`anthropic/claude-…`, `us.anthropic.claude-…`)
 * and of dated snapshots mis-captured as a version (`claude-3-5-sonnet-20241022`).
 */
declare function isAdaptiveThinkingModel(model: string | undefined): boolean;
/** Map a Responses `reasoning.effort` to an Anthropic `output_config.effort`
 *  value, or undefined when there is no direct equivalent (e.g. 'minimal'). */
declare function normalizeAnthropicEffort(effort: string | undefined): string | undefined;

interface TranslateResponseOptions$1 {
    responseId?: string;
    createdAt?: number;
    model?: string;
}
/** Convert a non-streaming Anthropic response into a Responses-API response. */
declare function translateResponse$1(body: AnthropicResponse, options?: TranslateResponseOptions$1): ResponsesResponse;
declare function mapOutputItems(content: AnthropicContentBlock[]): ResponsesOutputItem[];

interface TranslateStreamOptions$1 {
    model?: string;
    responseId?: string;
    createdAt?: number;
    requestMetadata?: ResponsesStreamMetadata$1;
}
interface ResponsesStreamMetadata$1 {
    temperature?: number;
    top_p?: number;
    tools?: unknown[];
    tool_choice?: unknown;
    store?: boolean;
    metadata?: Record<string, unknown>;
}
/**
 * Consume an Anthropic SSE stream and yield Responses-API SSE events.
 */
declare function translateStream$1(stream: ReadableStream<Uint8Array>, options?: TranslateStreamOptions$1): AsyncGenerator<ResponsesStreamEvent, void, void>;
/**
 * Consume an async iterable of parsed Anthropic events and yield Responses events.
 */
declare function translateAnthropicEvents(events: AsyncIterable<AnthropicStreamEvent> | Iterable<AnthropicStreamEvent>, options?: TranslateStreamOptions$1): AsyncGenerator<ResponsesStreamEvent, void, void>;

declare const index$2_isAdaptiveThinkingModel: typeof isAdaptiveThinkingModel;
declare const index$2_mapOutputItems: typeof mapOutputItems;
declare const index$2_normalizeAnthropicEffort: typeof normalizeAnthropicEffort;
declare const index$2_translateAnthropicEvents: typeof translateAnthropicEvents;
declare namespace index$2 {
  export { type ResponsesStreamMetadata$1 as ResponsesStreamMetadata, type TranslateRequestOptions$1 as TranslateRequestOptions, type TranslateRequestResult$1 as TranslateRequestResult, type TranslateResponseOptions$1 as TranslateResponseOptions, type TranslateStreamOptions$1 as TranslateStreamOptions, index$2_isAdaptiveThinkingModel as isAdaptiveThinkingModel, index$2_mapOutputItems as mapOutputItems, index$2_normalizeAnthropicEffort as normalizeAnthropicEffort, index$2_translateAnthropicEvents as translateAnthropicEvents, translateRequest$1 as translateRequest, translateResponse$1 as translateResponse, translateStream$1 as translateStream };
}

/**
 * OpenAI chat/completions API type definitions (subset used by the openai-chat upstream format).
 * Only the fields needed for request/response/stream translation are typed.
 */
interface OpenAiChatToolCall {
    id?: string;
    type?: 'function' | string;
    extra_content?: {
        google?: {
            thought_signature?: string;
            thought?: boolean;
            [key: string]: unknown;
        };
        [key: string]: unknown;
    };
    function?: {
        name?: string;
        arguments?: string | Record<string, unknown>;
    };
    thought_signature?: string;
}
interface OpenAiChatMessage {
    role: 'system' | 'user' | 'assistant' | 'tool' | string;
    content?: string | Array<{
        type: string;
        text?: string;
        [k: string]: unknown;
    }> | null;
    name?: string;
    tool_call_id?: string;
    tool_calls?: OpenAiChatToolCall[];
    reasoning_content?: string;
    [key: string]: unknown;
}
interface OpenAiChatFunctionTool {
    type: 'function';
    function: {
        name: string;
        description?: string;
        parameters?: Record<string, unknown>;
    };
}
interface OpenAiChatWebSearchTool {
    type: 'web_search';
    web_search: {
        enable: boolean;
        search_engine?: string;
        [k: string]: unknown;
    };
}
type OpenAiChatTool = OpenAiChatFunctionTool | OpenAiChatWebSearchTool | Record<string, unknown>;
type OpenAiChatToolChoice = 'auto' | 'required' | 'none' | {
    type: 'function';
    function: {
        name: string;
    };
} | Record<string, unknown>;
interface OpenAiChatRequest {
    model: string;
    messages: OpenAiChatMessage[];
    stream?: boolean;
    temperature?: number;
    top_p?: number;
    max_tokens?: number;
    tools?: OpenAiChatTool[];
    tool_choice?: OpenAiChatToolChoice;
    [key: string]: unknown;
}
interface OpenAiChatUsage {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
    prompt_tokens_details?: {
        cached_tokens?: number;
        [key: string]: unknown;
    };
    [key: string]: unknown;
}
interface OpenAiChatChoice {
    index?: number;
    message: {
        role?: string;
        content?: string | null;
        tool_calls?: OpenAiChatToolCall[];
        reasoning_content?: string;
        [key: string]: unknown;
    };
    finish_reason?: string;
    [key: string]: unknown;
}
interface OpenAiChatResponse {
    id?: string;
    object?: string;
    created?: number;
    model?: string;
    choices: OpenAiChatChoice[];
    usage?: OpenAiChatUsage;
    [key: string]: unknown;
}
interface OpenAiChatStreamDeltaToolCall {
    index: number;
    id?: string;
    type?: string;
    extra_content?: OpenAiChatToolCall['extra_content'];
    function?: {
        name?: string;
        arguments?: string | Record<string, unknown>;
    };
    thought_signature?: string;
}
interface OpenAiChatStreamDelta {
    role?: string;
    content?: string;
    reasoning_content?: string;
    extra_content?: OpenAiChatToolCall['extra_content'];
    tool_calls?: OpenAiChatStreamDeltaToolCall[];
    [key: string]: unknown;
}
interface OpenAiChatStreamChoice {
    index?: number;
    delta?: OpenAiChatStreamDelta;
    finish_reason?: string | null;
    [key: string]: unknown;
}
interface OpenAiChatStreamChunk {
    id?: string;
    object?: string;
    created?: number;
    model?: string;
    choices?: OpenAiChatStreamChoice[];
    usage?: OpenAiChatUsage;
    [key: string]: unknown;
}

interface TranslateRequestOptions {
    /** Default max tokens when not provided. */
    defaultMaxTokens?: number;
    /** If true, backfill reasoning_content on assistant tool-call messages. */
    backfillReasoning?: boolean;
    /** Placeholder string for backfilled reasoning_content. Defaults to '.'. */
    reasoningPlaceholder?: string;
    /** If true, strip `strict` from function tools (some upstreams reject it). */
    /** If true, drop image/file parts from user messages (e.g. DeepSeek text-only models). */
    dropImages?: boolean;
    /** Fallback signature for Gemini OpenAI histories that lack returned signatures. */
    fallbackThoughtSignature?: string;
    /** Recover a thought signature tunneled through the function_call `call_id`
     *  (see translateResponse/translateStream) and restore the clean call_id sent
     *  upstream. Must match the setting used when translating the response. */
    tunnelThoughtSignatureInCallId?: boolean;
}
interface TranslateRequestResult {
    request: OpenAiChatRequest;
}
/** Convert a Responses API request into an OpenAI Chat API request. */
declare function translateRequest(data: ResponsesRequest, options?: TranslateRequestOptions): TranslateRequestResult;

interface TranslateResponseOptions {
    responseId?: string;
    createdAt?: number;
    model?: string;
    /** Chat Completions tool list from the translated request — used to recover
     *  namespace when the upstream omits the "namespace." prefix in a tool call. */
    requestTools?: unknown[];
    /** Tunnel the Gemini thought signature through the function_call `call_id`
     *  (append it after a sentinel) for clients that drop `thought_signature`.
     *  translateRequest strips it back off. */
    tunnelThoughtSignatureInCallId?: boolean;
}
/** Convert an OpenAI Chat response into a Responses-API response. */
declare function translateResponse(body: OpenAiChatResponse, options?: TranslateResponseOptions): ResponsesResponse;

interface ResponsesStreamMetadata {
    temperature?: number;
    top_p?: number;
    tools?: unknown[];
    tool_choice?: unknown;
    store?: boolean;
    metadata?: Record<string, unknown>;
}
interface TranslateStreamOptions {
    model?: string;
    responseId?: string;
    createdAt?: number;
    requestMetadata?: ResponsesStreamMetadata;
    /** Tunnel the Gemini thought signature through the function_call `call_id`
     *  (append it after a sentinel) for clients that drop `thought_signature`.
     *  translateRequest strips it back off. */
    tunnelThoughtSignatureInCallId?: boolean;
}
/**
 * Consume an OpenAI-chat-style SSE stream and yield Responses-API SSE events.
 */
declare function translateStream(stream: ReadableStream<Uint8Array>, options?: TranslateStreamOptions): AsyncGenerator<ResponsesStreamEvent, void, void>;

type index$1_ResponsesStreamMetadata = ResponsesStreamMetadata;
type index$1_TranslateRequestOptions = TranslateRequestOptions;
type index$1_TranslateRequestResult = TranslateRequestResult;
type index$1_TranslateResponseOptions = TranslateResponseOptions;
type index$1_TranslateStreamOptions = TranslateStreamOptions;
declare const index$1_translateRequest: typeof translateRequest;
declare const index$1_translateResponse: typeof translateResponse;
declare const index$1_translateStream: typeof translateStream;
declare namespace index$1 {
  export { type index$1_ResponsesStreamMetadata as ResponsesStreamMetadata, type index$1_TranslateRequestOptions as TranslateRequestOptions, type index$1_TranslateRequestResult as TranslateRequestResult, type index$1_TranslateResponseOptions as TranslateResponseOptions, type index$1_TranslateStreamOptions as TranslateStreamOptions, index$1_translateRequest as translateRequest, index$1_translateResponse as translateResponse, index$1_translateStream as translateStream };
}

/**
 * Unified translation layer for Responses API.
 *
 * This module translates between OpenAI Responses API format and upstream
 * API formats. You specify the upstream API format (`anthropic` or
 * `openai-chat`) instead of a named provider.
 *
 * - `anthropic` → Anthropic Messages API
 * - `openai-chat` → OpenAI-compatible Chat Completions API (OpenAI, ZAI, etc.)
 */

type index_ResponsesRequest = ResponsesRequest;
type index_ResponsesResponse = ResponsesResponse;
type index_ResponsesStreamEvent = ResponsesStreamEvent;
declare namespace index {
  export { type index_ResponsesRequest as ResponsesRequest, type index_ResponsesResponse as ResponsesResponse, type index_ResponsesStreamEvent as ResponsesStreamEvent, index$2 as anthropic, index$1 as openai };
}

/**
 * SSE helpers.
 *
 * - `parseSseStream`: Consume a ReadableStream of bytes and yield parsed
 *   `{ event, data }` pairs.  Works in both browser (fetch body) and Node 18+.
 * - `encodeSseEvent`: Serialize a `{ event, data }` pair to the SSE wire format.
 */
interface SseMessage {
    event?: string;
    data: string;
}
declare function parseSseStream(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseMessage, void, void>;
declare function encodeSseEvent(event: string, data: unknown): string;

export { type AnthropicContentBlock, type AnthropicMessage, type AnthropicRequest, type AnthropicResponse, type AnthropicStreamEvent, type AnthropicTool, type AnthropicToolChoice, type AnthropicUsage, type CacheStats, type CreateResponsesFetchOptions, type OpenAiChatMessage, type OpenAiChatRequest, type OpenAiChatResponse, type OpenAiChatStreamChunk, type OpenAiChatTool, type OpenAiChatToolCall, type ResponsesContentPart, type ResponsesInputItem, type ResponsesOutputFunctionCall, type ResponsesOutputItem, type ResponsesOutputMessage, type ResponsesOutputReasoning, type ResponsesRequest, type ResponsesResponse, type ResponsesStreamEvent, type ResponsesTool, type ResponsesToolChoice, type ResponsesUsage, type SseMessage, type UpstreamFormat, createResponsesFetch, encodeSseEvent, parseSseStream, index as translate };
