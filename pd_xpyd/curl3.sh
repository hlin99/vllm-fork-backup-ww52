jq -Rs \
  --arg model "/mnt/disk2/hf_models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2/" \
  --arg instr "请用5000字左右总结一下这段文字内容，列出重点信息" \
  '{
    model: $model,
    prompt: "\($instr)\(.)",
    max_tokens: 5000,
    temperature: 0.3
  }' hlin999.txt \
| curl http://10.239.129.9:8868/v1/completions \
    -H "Content-Type: application/json" \
    -d @-

