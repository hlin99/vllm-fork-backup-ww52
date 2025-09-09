curl http://10.239.129.9:8868/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/mnt/disk2/hf_models/DeepSeek-R1-G2-static/",
        "prompt": "The expression $2\\cdot 3 \\cdot 4\\cdot 5+1$ is equal to 121, since multiplication is carried out before addition. However, we can obtain values other than 121 for this expression if we are allowed to change it by inserting parentheses. For example, we can obtain 144 by writing \\[\n(2\\cdot (3\\cdot 4)) \\cdot (5+1) = 144.\n\\]In total, how many values can be obtained from the expression $2\\cdot 3\\cdot 4 \\cdot 5 + 1$ by inserting parentheses? (Note that rearranging terms is not allowed, only inserting parentheses).\nPlease reason step by step, and put your final answer within \\boxed{}.The expression $2\\cdot 3 \\cdot 4\\cdot 5+1$ is equal to 121, since multiplication is carried out before addition. However, we can obtain values other than 121 for this expression if we are allowed to change it by inserting parentheses. For example, we can obtain 144 by writing \\[\n(2\\cdot (3\\cdot 4)) \\cdot (5+1) = 144.\n\\]In total, how many values can be obtained from the expression $2\\cdot 3\\cdot 4 \\cdot 5 + 1$ by inserting parentheses? (Note that rearranging terms is not allowed, only inserting parentheses).\nPlease reason step by step, and put your final answer within \\boxed{}.",
        "max_tokens": 30000,
        "temperature": 0
    }'

