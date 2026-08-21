# @codeproxy/core

> **English** → [README.md](./README.md)

**@codeproxy/core** 是一个零依赖的 TypeScript 库，将上游 LLM 的 Chat Completions 或 Anthropic Messages 格式**转换为 OpenAI Responses API 格式**，让 Codex、Claude Code 等 AI 编码代理能够直接调用任意语言模型。

目标：**让任何语言模型都能直接在 Codex 或 Claude Code 中使用**，无论上游提供商使用什么 API 格式。

## 快速开始

```ts
import OpenAI from 'openai';
import { createResponsesFetch } from '@codeproxy/core';

const client = new OpenAI({
  fetch: createResponsesFetch({
    upstreamFormat: 'openai-chat',
    baseUrl: 'https://api.deepseek.com/v1',
    defaultHeaders: {
      authorization: `Bearer ${process.env.DEEPSEEK_API_KEY}`,
    },
  }),
});

const response = await client.responses.create({
  model: 'deepseek-v4-flash',
  input: 'Hello!',
});
```

## 工作原理

```
上游 API (Chat Completions / Anthropic Messages)
        │
        ▼  createResponsesFetch 自动转换
        │
OpenAI Responses API ←── 供 Codex / Claude Code 消费
```

> **CLI** → [`@codeproxy/cli`](https://github.com/codeproxy-ai/cli) — 本地代理服务器，让 Codex / Claude Code 使用任意模型

## 安装

```bash
npm install @codeproxy/core
```

## API

### `createResponsesFetch(options)`

创建一个 `fetch` 包装器，自动将 Responses API 流量转换到配置的上游格式。

| 参数 | 类型 | 说明 |
|---|---|---|
| `upstreamFormat` | `'anthropic'` \| `'openai-chat'` | 上游 API 格式（省略时从 baseUrl 推断） |
| `baseUrl` | `string` | 上游端点 URL |
| `apiVersion` | `string` | 覆盖 Anthropic 版本头 |
| `model` | `string` | 为所有请求覆盖模型名 |
| `defaultHeaders` | `Record<string, string>` | 发送给上游的额外请求头 |
| `dropImages` | `boolean` | 从消息中移除图片部分（纯文本模型） |
| `timeoutMs` | `number` | 上游请求超时 |
| `onCacheStats` | `(stats) => void` | 接收缓存使用统计 |
| `fallbackThoughtSignature` | `string` | Gemini OpenAI 兼容工具调用历史的兜底 thought signature |
| `tunnelThoughtSignatureInCallId` | `boolean` | 把 Gemini thought signature 编码进 function-call 的 `call_id`，让它在会丢弃 `thought_signature` 字段的客户端里存活 |
| `fallbackUpstream` | `object` | 当 `dropImages: true` 且请求包含图片时，自动切换到该上游（如支持视觉的模型） |

Gemini OpenAI 兼容工具调用会在 Responses function-call item 中保留 `thought_signature`。迁移旧历史时，如果历史里没有上游返回的 signature，可以传入 `fallbackThoughtSignature`，让下一次上游工具调用请求继续带上 `extra_content.google.thought_signature`。

有些客户端（如 codex-ts，其协议镜像 codex-rs）在 function-call item 上只往返 `{ type, call_id, name, arguments }`，会丢掉上游返回的 `thought_signature`，导致 Gemini 跨轮次思维链断裂。开启 `tunnelThoughtSignatureInCallId` 后，会在响应侧把 signature 追加进 `call_id`（以 `~gts~` 分隔），并在下一次请求侧拆回——客户端只存一个不透明字符串、无需改动，最终发往上游的仍是干净的 `call_id`。请求与响应两侧的翻译需使用相同设置。

### 转换器

底层转换器也按命名空间导出：

- `anthropic.translateRequest` / `anthropic.translateResponse` / `anthropic.translateStream`
- `openai.translateRequest` / `openai.translateResponse` / `openai.translateStream`
- `translate.*` — 统一重新导出

### 类型

所有 Responses API、Anthropic 和 OpenAI Chat 类型均从包根目录导出。

### 工具函数

- `parseSseStream(stream)` — 消费 `ReadableStream` → 解析后的 SSE 消息
- `encodeSseEvent(event, data)` — 序列化 SSE 事件为线格式
- `makeId(prefix)` — 生成单调递增的 ID

## 协议

MIT
