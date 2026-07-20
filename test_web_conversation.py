"""
Test ChatGPT web /backend-api/conversation endpoint using curl_cffi
to bypass Cloudflare TLS fingerprint detection.
"""

import json
import uuid
from pathlib import Path
from curl_cffi import requests


def main():
    # Load auth token
    auth_path = Path.home() / ".codex" / "auth.json"
    with open(auth_path) as f:
        auth = json.load(f)
    
    tokens = auth["tokens"]
    access_token = tokens["access_token"]
    account_id = tokens.get("account_id", "")
    
    print(f"Token loaded, account_id: {account_id}")
    
    # ChatGPT web conversation endpoint
    url = "https://chatgpt.com/backend-api/conversation"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "chatgpt-account-id": account_id,
        "oai-language": "en-US",
    }
    
    # Web conversation request format
    message_id = str(uuid.uuid4())
    parent_id = str(uuid.uuid4())
    
    body = {
        "action": "next",
        "messages": [
            {
                "id": message_id,
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": ["Say hi in one word"]},
                "metadata": {},
            }
        ],
        "parent_message_id": parent_id,
        "model": "gpt-4o",
        "timezone_offset_min": -480,
        "history_and_training_disabled": False,
        "conversation_mode": {"kind": "primary_assistant"},
        "force_paragen": False,
        "force_paragen_model_slug": "",
        "force_nulligen": False,
        "force_rate_limit": False,
        "websocket_request_id": str(uuid.uuid4()),
    }
    
    print(f"\nRequest URL: {url}")
    print(f"Model: {body['model']}")
    print(f"\n{'='*60}")
    
    # Use curl_cffi with Chrome impersonation to bypass Cloudflare
    resp = requests.post(
        url,
        json=body,
        headers=headers,
        impersonate="chrome",
        stream=True,
    )
    
    print(f"Status: {resp.status_code}")
    
    if resp.status_code != 200:
        print(f"Error: {resp.text[:500]}")
        return
    
    # Parse SSE stream
    full_text = ""
    msg_count = 0
    for raw_line in resp.iter_lines():
        line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
        line = line.strip()
        if not line:
            continue
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                print(f"\n[DONE]")
                break
            try:
                payload = json.loads(data)
                msg_count += 1
                
                # Show first few raw events
                if msg_count <= 3:
                    print(f"\n[Event #{msg_count}] keys={list(payload.keys())}")
                    if "message" in payload:
                        msg = payload["message"]
                        print(f"  author={msg.get('author')}")
                        print(f"  content_type={msg.get('content',{}).get('content_type')}")
                        print(f"  status={msg.get('status')}")
                
                # Extract content
                message = payload.get("message", {})
                if message:
                    content = message.get("content", {})
                    parts = content.get("parts", [])
                    if parts and isinstance(parts[0], str):
                        new_text = parts[0]
                        if len(new_text) > len(full_text):
                            delta = new_text[len(full_text):]
                            print(delta, end="", flush=True)
                            full_text = new_text
            except json.JSONDecodeError:
                pass
    
    print(f"\n{'='*60}")
    print(f"Full response: {full_text!r}")
    print(f"Total events: {msg_count}")


if __name__ == "__main__":
    main()
