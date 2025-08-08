jq -Rs \
  --arg model "/mnt/disk2/hf_models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2/" \
  --arg instr "请总结一下文件的内容：" \
  '{
    model: $model,
    messages: [
      {role: "user", content: "\($instr)\(.)"}
    ],
    max_tokens: 5000,
    temperature: 0
  }' output.txt > debug.json

cat debug.json

