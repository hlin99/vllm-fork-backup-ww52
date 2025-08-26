import requests

# vLLM API 基础 URL
BASE_URL = "http://10.239.129.9:8100"

# Endpoint 列表
endpoints = [
    ("/openapi.json", ["GET", "HEAD"]),
#    ("/docs", ["GET", "HEAD"]),
#    ("/docs/oauth2-redirect", ["GET", "HEAD"]),
#    ("/redoc", ["GET", "HEAD"]),
#    ("/health", ["GET"]),
#    ("/ping", ["GET", "POST"]),
#    ("/tokenize", ["POST"]),
#    ("/detokenize", ["POST"]),
#    ("/v1/models", ["GET"]),
#    ("/version", ["GET"]),
#    ("/v1/chat/completions", ["POST"]),
#    ("/v1/completions", ["POST"]),
#    ("/v1/embeddings", ["POST"]),
#    ("/pooling", ["POST"]),
#    ("/score", ["POST"]),
#    ("/v1/score", ["POST"]),
#    ("/v1/audio/transcriptions", ["POST"]),
#    ("/rerank", ["POST"]),
#    ("/v1/rerank", ["POST"]),
#    ("/v2/rerank", ["POST"]),
#    ("/invocations", ["POST"]),
]

sample_payload = {
    "/ping": {},
    "/tokenize": {"text": "Hello world"},
    "/detokenize": {"tokens": [15496, 995]},
    "/v1/chat/completions": {"model": "dummy", "messages": [{"role": "user", "content": "Hello"}]},
    "/v1/completions": {"model": "dummy", "prompt": "Hello", "max_tokens": 5},
    "/v1/embeddings": {"model": "dummy", "input": "Hello"},
    "/pooling": {"vectors": [[0.1, 0.2, 0.3]]},
    "/score": {"predictions": [0.1, 0.9]},
    "/v1/score": {"predictions": [0.1, 0.9]},
    "/v1/audio/transcriptions": {"audio": "dummy_audio_data"},
    "/rerank": {"queries": ["hello"], "documents": ["world"]},
    "/v1/rerank": {"queries": ["hello"], "documents": ["world"]},
    "/v2/rerank": {"queries": ["hello"], "documents": ["world"]},
    "/invocations": {"task": "echo", "input": "Hello"}
}

def test_endpoint(route, methods):
    for method in methods:
        url = BASE_URL + route
        try:
            if method == "POST":
                payload = sample_payload.get(route, {})
                resp = requests.post(url, json=payload, timeout=10)
            else:
                resp = requests.request(method, url, timeout=5)

            print(f"[{method}] {route} -> Status: {resp.status_code}")

            # 只有 GET / POST 才尝试解析 JSON
            if method in ["GET", "POST"] and resp.headers.get("content-type", "").startswith("application/json"):
                try:
                    data = resp.json()
                    print("JSON keys:", list(data.keys()))
                except Exception as e:
                    print("JSON解析失败:", e)
            else:
                print(resp.text[:200])  # 输出前200字符

        except Exception as e:
            print(f"[{method}] {route} -> Exception: {e}")

if __name__ == "__main__":
    for route, methods in endpoints:
        test_endpoint(route, methods)
