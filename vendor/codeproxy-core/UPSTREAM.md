# Vendored upstream

- Project: `@codeproxy/core`
- Repository: https://github.com/codeproxy-ai/core
- Version: `0.1.25`
- Commit: `4765d937d927b3f5588d18808be2c2245fcdca7e`
- Package-declared license: MIT
- Vendored on: 2026-08-18

The checked-in `dist/` directory is built from the commit above without local
source changes. The upstream repository at this commit does not contain a
standalone LICENSE file; its `package.json` declares `MIT`.

Codex-shim uses the OpenAI Responses API <-> OpenAI Chat Completions
translation layer for DeepSeek compatibility. DeepSeek Web-specific malformed
tool-call recovery remains in `vendor/deepseek-web-api` and feeds standard Chat
Completions chunks into this translator.
