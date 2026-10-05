#!/usr/bin/env python3
"""North-Mini-Code (cohere2_moe) live end-to-end on an MLX pack.

Boots the pack, then checks: the model is advertised, a short greedy answer with thinking off,
thinking on by default (reasoning split from content, no raw markers), a prompt past the 4096-key
sliding window, a tool call (non-streaming and streaming), and a tool-result round trip. The fused
router engagement is read from the server's own log line. SKIPs without the pack.

  NORTH_MINI_MODEL=<pack dir> python3 tests/test_north_mini.py [port]
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

MODEL = os.environ.get("NORTH_MINI_MODEL", "")
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 11413
BIN = os.environ.get("MLX_SERVE_BIN", "./zig-out/bin/mlx-serve")
LOG = os.path.expanduser(f"~/claude-tmp/north-mini-live/server-{PORT}.log")
URL = f"http://127.0.0.1:{PORT}"
MARKERS = ("<|START", "<|END", "<think>", "</think>")

if not os.path.isfile(os.path.join(MODEL, "config.json")):
    print(f"SKIP: no pack at {MODEL!r} (set NORTH_MINI_MODEL)")
    sys.exit(0)


def call(path, body=None, timeout=600):
    req = urllib.request.Request(
        URL + path, json.dumps(body).encode() if body else None, {"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chat(messages, **extra):
    body = {"model": "n", "messages": messages, "temperature": 0, "max_tokens": 400, **extra}
    return call("/v1/chat/completions", body)["choices"][0]


def stream(messages, **extra):
    """Returns (content, reasoning, tool_call_deltas, finish_reason) of a streamed answer."""
    body = {"model": "n", "messages": messages, "temperature": 0, "max_tokens": 600, "stream": True, **extra}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    content = reasoning = ""
    calls, finish = [], None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            choices = json.loads(line[6:])["choices"]
            if not choices:
                continue
            d = choices[0]["delta"]
            finish = choices[0].get("finish_reason") or finish
            content += d.get("content") or ""
            reasoning += d.get("reasoning_content") or d.get("reasoning") or ""
            if d.get("tool_calls"):
                calls.append(d["tool_calls"])
    return content, reasoning, calls, finish


os.makedirs(os.path.dirname(LOG), exist_ok=True)
server = subprocess.Popen(
    [BIN, "--model", MODEL, "--serve", "--host", "127.0.0.1", "--port", str(PORT),
     "--ctx-size", "16384", "--prefill-chunk", "1024"],
    stdout=open(LOG, "w"), stderr=subprocess.STDOUT,
)
passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    passed, failed = passed + bool(ok), failed + (not ok)
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)


TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
ASK = [{"role": "user", "content": "What is the weather in Berlin right now? Use the tool."}]

try:
    for _ in range(240):
        try:
            if call("/health", timeout=2).get("status") == "ok":
                break
        except Exception:
            time.sleep(1)
    else:
        sys.exit(f"server did not come up, see {LOG}")

    ids = [m["id"] for m in call("/v1/models")["data"]]
    check("[1] model advertised", len(ids) == 1, str(ids))

    c = chat([{"role": "user", "content": "What is 17 + 25? Answer with just the number."}], reasoning_effort="none")
    check("[2] thinking off answers, no reasoning text", c["message"]["content"].strip() == "42"
          and not c["message"].get("reasoning_content"), repr(c["message"]["content"]))

    c = chat([{"role": "user", "content": "What is 17 + 25? Answer with just the number."}])
    m = c["message"]
    check("[3] thinking on by default: reasoning split from the answer",
          m["content"].strip() == "42" and bool(m.get("reasoning_content")) and
          not any(x in m["content"] + m["reasoning_content"] for x in MARKERS), repr(m["content"]))

    items = " ".join(f"Item {i}: the value is {i * 7 % 13}." for i in range(520))  # ~5.5k tokens, past the window
    c = chat([{"role": "user", "content": items + "\n\nWhat is the value of Item 3? Answer with just the number."}],
             reasoning_effort="none", max_tokens=16)
    check("[4] retrieval past the sliding window", c["message"]["content"].strip() == "8", repr(c["message"]["content"]))

    c = chat(ASK, tools=TOOLS, tool_choice="auto")
    calls = c["message"].get("tool_calls") or []
    check("[5] tool call", c["finish_reason"] == "tool_calls" and len(calls) == 1
          and calls[0]["function"]["name"] == "get_weather"
          and json.loads(calls[0]["function"]["arguments"]).get("city") == "Berlin")

    content, reasoning, deltas, finish = stream(ASK, tools=TOOLS)
    check("[6] streamed tool call, nothing leaked", finish == "tool_calls" and len(deltas) == 1 and not content
          and not any(x in content + reasoning for x in MARKERS), f"content={content!r}")

    c = chat(ASK + [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
         "function": {"name": "get_weather", "arguments": "{\"city\": \"Berlin\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\": 11, \"conditions\": \"light rain\"}"},
    ], tools=TOOLS)
    answer = (c["message"].get("content") or "")
    check("[7] tool result round trip", c["finish_reason"] == "stop" and "11" in answer, repr(answer[:80]))
finally:
    server.terminate()
    server.wait(timeout=60)

log = open(LOG).read()
check("[8] fused router engaged", "fused router kernel engaged: mode=logit_bias" in log)
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
