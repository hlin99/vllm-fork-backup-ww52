curl http://10.239.129.9:8868/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "/mnt/disk2/hf_models/DeepSeek-R1-G2-static/",
        "messages": [
        	{"role": "system", "content": ""},
        ],
        "max_tokens": 1000,
        "temperature": 0
      }'

