# Codex-Shim 部署文档

将 Codex CLI/Desktop 的请求通过 ChatGPT Plus 订阅代理，免去 platform.openai.com API 单独付费。

## 1. 前置条件

- Python 3.12+
- ChatGPT Plus 账号
- Codex CLI 已安装 (`npm install -g @openai/codex`)

## 2. 安装

```bash
cd /opt/codes/codex-shim
pip install -e .
```

## 3. 认证配置

### 3.1 获取 Token

```bash
codex login --device-auth
```

会给出一个 URL + 验证码，用浏览器打开 URL，登录 ChatGPT 账号确认。完成后 `~/.codex/auth.json` 自动写入 token。

Token 有效期 **10 天**，过期后重新执行此命令即可。

### 3.2 auth.json 结构

```json
{
  "auth_mode": "chatgpt",
  "tokens": {
    "access_token": "ey...",
    "refresh_token": "...",
    "account_id": "你的account-id"
  }
}
```

### 3.3 Codex 指向 Shim

```bash
codex-shim enable
```

或手动编辑 `~/.codex/config.toml`：

```toml
model_provider = "codex_shim"
model = "gpt-5.5"

[codex_shim]
base_url = "http://127.0.0.1:8765/v1"
wire_api = "responses"
```

## 4. 运行

### 4.1 前台运行（调试用）

```bash
cd /opt/codes/codex-shim
PYTHONUNBUFFERED=1 python3 -m codex_shim.server --host 127.0.0.1 --port 8765
```

### 4.2 健康检查

```bash
curl http://127.0.0.1:8765/health
# 返回: {"ok": true, "models": 7, "chatgpt_passthrough": true, ...}
```

### 4.3 手动测试请求

```bash
curl -X POST http://127.0.0.1:8765/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test" \
  -d '{"model":"gpt-5.5","input":[{"role":"user","content":[{"type":"input_text","text":"Hello"}]}],"stream":true,"store":false}'
```

## 5. 守护进程

### 5.1 shim-guard.sh（当前会话保活）

位置：`/opt/codes/codex-shim/shim-guard.sh`

```bash
#!/bin/bash
cd /opt/codes/codex-shim
export PYTHONPATH="/opt/codes/codex-shim:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

while true; do
    echo "[shim-guard] Starting codex-shim at $(date)" >> /tmp/shim.log
    python3 -m codex_shim.server --host 127.0.0.1 --port 8765 >> /tmp/shim.log 2>&1
    EXIT_CODE=$?
    echo "[shim-guard] codex-shim exited with code $EXIT_CODE at $(date)" >> /tmp/shim.log
    sleep 2
done
```

启动：

```bash
nohup /opt/codes/codex-shim/shim-guard.sh &
```

日志：`/tmp/shim.log`

### 5.2 s6-overlay 服务（容器重启自动拉起）

服务文件：`/etc/s6-overlay/s6-rc.d/svc-codex-shim/run`

```bash
#!/usr/bin/with-contenv bash
cd /opt/codes/codex-shim
export PYTHONPATH="/opt/codes/codex-shim:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
exec python3 -m codex_shim.server --host 127.0.0.1 --port 8765
```

注册文件：

```
/etc/s6-overlay/s6-rc.d/svc-codex-shim/type        → 内容: "longrun"
/etc/s6-overlay/s6-rc.d/user/contents.d/svc-codex-shim  → 空文件
```

s6 作为 supervisor，shim 崩溃会自动重启。

## 6. 已修复的问题

| 问题 | 原因 | 修复 |
|------|------|------|
| "Store must be set to false" | ChatGPT Codex 端点要求 store=false | `_sanitize_chatgpt_passthrough_body` 强制设 store=False |
| "Stream must be set to true" | ChatGPT Codex 端点要求 stream=true | 同上强制设 stream=True |
| "Unknown parameter: store" (compact) | compact 端点不接受 store 参数 | compact 路径 pop("store") |
| 请求慢 2-3 秒 | 每次新建 TCP+TLS 连接 | 持久化 ClientSession 连接池 |
| 空闲后 500 错误 | 连接池 TCP 被服务端断开 | 捕获连接异常，自动 reset session 重试 |
| "high demand" 临时错误 | ChatGPT 后端限流 (429/503/529) | 指数退避重试，最多 5 次 |

## 7. Token 过期提醒

定时任务每天 09:00 UTC 检查 token 剩余时间：

- 脚本：`~/.hermes/scripts/check_codex_token.sh`
- 剩余 > 48h → 静默
- 剩余 < 48h → 提醒执行 `codex login --device-auth`
- 已过期 → 紧急提醒

## 8. 注意事项

- 仅 **gpt-5.5** 可用（ChatGPT Plus Codex 通道只支持此模型）
- Plus 订阅本身有速率限制（如 GPT-4o 80条/3h），密集使用会触发 429
- 此方式违反 OpenAI ToS，个人轻度使用风险较低
- `sock_connect` 超时 30 秒，`keepalive_timeout` 120 秒
- 连接池上限 20 个并发连接，DNS 缓存 600 秒

## 9. 文件路径一览

```
/opt/codes/codex-shim/                  # 项目根目录
├── codex_shim/server.py                # 主服务（含连接池、重试逻辑）
├── shim-guard.sh                       # 保活脚本
├── .codex-shim/config.toml             # shim 自身配置
└── .codex-shim/custom_model_catalog.json  # 模型目录（7个模型）

~/.codex/auth.json                      # 认证 token
~/.codex/config.toml                    # Codex 配置（指向 shim）
~/.hermes/scripts/check_codex_token.sh  # token 过期检查脚本

/etc/s6-overlay/s6-rc.d/svc-codex-shim/ # s6 服务定义
/tmp/shim.log                           # 运行日志
```
