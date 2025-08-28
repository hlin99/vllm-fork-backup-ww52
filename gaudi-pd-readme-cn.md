# 在G2D server PD分离运行DeepSeek-R1

## 准备模型
本文的示例假定已经转换好的DeepSeek-R1模型放在/mnt/disk3/hf_models/DeepSeek-R1-G2目录，/mnt/disk3在启动docker容器被映射为/data。所以在docker容器中，模型所在目录为/data/hf_modes/DeepSeek-R1-G2

如果还没有准备好模型，可以在参考下面两部下载并转换模型，这两步需要在docker容器启动之后在容器中完成

### 下载模型
可以在[HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1) and [ModelScope](https://www.modelscope.cn/deepseek-ai/DeepSeek-R1)下载模型
```bash
sudo apt install git-lfs
git-lfs install

# 选项1: 从HuggingFace下载
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1 /data/hf_models/DeepSeek-R1
# 选项2: 从ModelScope下载
git clone https://www.modelscope.cn/deepseek-ai/DeepSeek-R1 /data/hf_models/DeepSeek-R1
```

### 转换模型
原始模型需要转换才能在Gaudi2上运行，假设原始模型在/data/hf_models/DeepSeek-R1目录下，使用以下命令完成转换，转好的模型在/data/hf_models/DeepSeek-R1-G2目录下。请确保磁盘有足够的空间(>650GB)。如果磁盘性能正常，转换过程大概需要15分钟

```bash
cd /ws/vllm-fork
pip install compress_pickle torch safetensors numpy --extra-index-url https://download.pytorch.org/whl/cpu

python scripts/convert_block_fp8_to_channel_fp8.py --model_path /data/hf_models/DeepSeek-R1 --qmodel_path /data/hf_models/DeepSeek-R1-G2 --input_scales_path scripts/DeepSeek-R1-BF16-w8afp8-static-no-ste_input_scale_inv.pkl.gz
```
转换完成可以看到如下输出

```bash
INFO:__main__:[160/163] Saving 649 tensors to /data/hf_models/DeepSeek-R1-G2/model-00160-of-000163.safetensors
100%|█████████████████████████████████████████████████████████████████████████████████████████████| 163/163 [11:30<00:00,  4.24s/it]
Saving tensor mapping to /data/hf_models/DeepSeek-R1-G2/model.safetensors.index.json
Conversion is completed.
```

## Host OS 设置
对所有服务器
```bash
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
echo always > /sys/kernel/mm/transparent_hugepage/enabled
echo 32768 > /proc/sys/vm/nr_hugepages
cat /proc/meminfo | grep Huge
cat /proc/sys/vm/nr_hugepages
```

## 查看RDMA
可以使用ibdev2netdev (host OS) 查看RDMA设备是否可用, 下文中mooncake_3fg5[6].json中device_name对应的设备应该是Up
```bash
ibdev2netdev
mlx5_0 port 1 ==> ens108np0 (Up)
mlx5_3 port 1 ==> ens109np0 (Up)
mlx5_4 port 1 ==> ens110np0 (Up)
mlx5_5 port 1 ==> ens111np0 (Up)
mlx5_6 port 1 ==> ens112np0 (Up)
mlx5_7 port 1 ==> ens113np0 (Up)
mlx5_8 port 1 ==> ens114np0 (Up)
mlx5_9 port 1 ==> ens115np0 (Up)
```
## 默认设置
默认设置中，model_len (8k) 和 max_num_seqs (8) 较小，不能在高并发请求的环境下工作。

## Web代理设置
如果你的环境变量中设置了Web代理，需要把所有的prefill节点和decoding节点放在no_proxy列表中

## 1P1D PD分离环境准备
### 准备prefill节点（同时也是metadata etcd服务器，以及mooncake master）
上传docker镜像到prefille节点，使用docker load命令载入

使用以下命令启动docker容器
```bash
docker run -it --name ds-pd -d --runtime=habana -e HABANA_VISIBLE_DEVICES=all --device=/dev:/dev -v /dev:/dev -v /mnt/disk3:/data -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --cap-add SYS_PTRACE --cap-add=CAP_IPC_LOCK --ulimit memlock=-1:-1 --net=host --ipc=host baidu-gaudi-pytorch2.6.0-vllm-ds-pd-aug15:1.21.2-76 /bin/bash
```
进入/ws/vllm-fork/pd_xpyd目录，修改示例mooncake_3fg5.json完成设置

如果RDMA设备不可用，把协议设置为tcp

示例mooncake_3fg5.json中，prefill节点的IP地址为192.168.1.105

在这个示例中，metadata_server (etcd), master_server_address 和prefill节点是一样的

配置文件名称中的一部分，示例mooncake_3fg5.json里的3fg5，会作为命令行参数传给启动脚本

如果没有使用3fg5作为prefille配置的名字，需要编辑1p_start_prefilll.sh, 在第16行把3fg5替换为您指定的名字

如果模型没有在默认的/data/hf_models/DeepSeek-R1-G2，需要编辑pd_env.sh指定模型所在目录

使用以下命令启动prefill服务器
```bash
bash 1p_start_prefill.sh 3fg5
```
可以使用ps ax命令查看etcd和mooncake master是否已经启动

### 准备decoding节点
启动容器(步骤和prefill节点一样)

进入/ws/vllm-fork/pd_xpyd，修改示例mooncake_3fg6.json完成设置

如果RDMA设备不可用，把协议设置为tcp

示例mooncake_3fg6.json中，decoding节点的IP地址为192.168.1.106

metadata_server (etcd), master_server_address 和prefill节点是一样的

使用以下命令启动decoding服务器
```bash
bash 1d_start_decode.sh 3fg6
```

### 启动 PD proxy 服务器
代理服务器运行在prefill节点，编辑/ws/vllm-fork/pd_xpyd目录下的xpyd_start_proxy.sh，指定prefille server（变量：PREFILL_IPS）和decoding server（变量：DECODE_IPS）的IP地址。在示例中，地址分别是 192.168.1.105和192.168.1.106

使用以下命令启动PD proxy服务器
```bash
cd /ws/vllm-fork
bash pd_xpyd/xpyd_start_proxy.sh 1 1 8 false
```

### 验证
所有服务器启动后有如下输出
```bash
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```
在prefill节点运行以下命令测试服务器
```bash
cd /ws/vllm-fork
python3 benchmarks/benchmark_serving.py --backend vllm --model /data/hf_models/DeepSeek-R1-G2 --dataset-name sonnet --request-rate inf --host localhost --port 8868 --sonnet-input-len 3500 --sonnet-output-len 1000 --sonnet-prefix-len 100 --trust-remote-code --max-concurrency 1024 --num-prompts 1 --ignore-eos --burstiness 1000 --dataset-path benchmarks/sonnet.txt --save-result
```

期望输出如下
```bash
NFO 08-15 02:37:34 __init__.py:199] Automatically detected platform hpu.
Namespace(backend='vllm', base_url=None, host='localhost', port=8868, endpoint='/v1/completions', dataset=None, dataset_name='sonnet', dataset_path='benchmarks/sonnet.txt', max_concurrency=1024, model='/data/hf_models/DeepSeek-R1-G2', tokenizer=None, best_of=1, use_beam_search=False, num_prompts=1, logprobs=None, request_rate=inf, burstiness=1000.0, seed=0, trust_remote_code=True, disable_tqdm=False, profile=False, save_result=True, metadata=None, result_dir=None, result_filename=None, ignore_eos=True, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, sonnet_input_len=3500, sonnet_output_len=1000, sonnet_prefix_len=100, sharegpt_output_len=None, random_input_len=1024, random_output_len=128, random_range_ratio=1.0, random_prefix_len=0, hf_subset=None, hf_split=None, hf_output_len=None, tokenizer_mode='auto', served_model_name=None, lora_modules=None)
Starting initial single prompt test run...
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1000.0 (Gamma distribution)
Maximum request concurrency: 1024
100%|█████████████████████████████████████████████| 1/1 [00:22<00:00, 22.47s/it]
============ Serving Benchmark Result ============
Successful requests:                     1         
Benchmark duration (s):                  22.47     
Total input tokens:                      3146      
Total generated tokens:                  1000      
Request throughput (req/s):              0.04      
Output token throughput (tok/s):         44.50     
Total Token throughput (tok/s):          184.49    
---------------Time to First Token----------------
Mean TTFT (ms):                          3043.48   
Median TTFT (ms):                        3043.48   
P99 TTFT (ms):                           3043.48   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          19.45     
Median TPOT (ms):                        19.45     
P99 TPOT (ms):                           19.45     
---------------Inter-token Latency----------------
Mean ITL (ms):                           19.45     
Median ITL (ms):                         19.30     
P99 ITL (ms):                            24.86     
==================================================
```

## 1P2D PD分离环境准备

### 准备prefill节点（同时也是metadata etcd服务器，以及mooncake master）
上传docker镜像到prefille节点，使用docker load命令载入

使用以下命令启动docker容器
```bash
docker run -it --name ds-pd-1p2d -d --runtime=habana -e HABANA_VISIBLE_DEVICES=all --device=/dev:/dev -v /dev:/dev -v /mnt/disk3:/data -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --cap-add SYS_PTRACE --cap-add=CAP_IPC_LOCK --ulimit memlock=-1:-1 --net=host --ipc=host baidu-gaudi-pytorch2.6.0-vllm-ds-pd-aug15:1.21.2-76 /bin/bash
```
进入/ws/vllm-fork/pd_xpyd目录，修改更新RECIPE_CACHE目录
修改dp_p_env.sh， 将
```bash
export PT_HPU_RECIPE_CACHE_CONFIG=/ws/cache-pd-p,false,16384
```
更新为
```bash
export PT_HPU_RECIPE_CACHE_CONFIG=/ws/cache-pd-1p2d-p,false,16384
```

修改dp_d_env.sh， 将
```bash
export PT_HPU_RECIPE_CACHE_CONFIG=/ws/cache-pd-d,false,16384
```
更新为
```bash
export PT_HPU_RECIPE_CACHE_CONFIG=/ws/cache-pd-1p2d-d,false,16384
```

以下的设置与PD 1P1D一样
进入/ws/vllm-fork/pd_xpyd目录，修改示例mooncake_3fg5.json完成设置

如果RDMA设备不可用，把协议设置为tcp

示例mooncake_3fg5.json中，prefill节点的IP地址为192.168.1.105

在这个示例中，metadata_server (etcd), master_server_address 和prefill节点是一样的

配置文件名称中的一部分，示例mooncake_3fg5.json里的3fg5，会作为命令行参数传给启动脚本

如果没有使用3fg5作为prefille配置的名字，需要编辑1p_start_prefilll.sh, 在第16行把3fg5替换为您指定的名字

如果模型没有在默认的/data/hf_models/DeepSeek-R1-G2，需要编辑pd_env.sh指定模型所在目录

使用以下命令启动prefill服务器
```bash
bash 1p_start_prefill.sh 3fg5
```
可以使用ps ax命令查看etcd和mooncake master是否已经启动


### 准备decoding节点1
启动容器(步骤和prefill节点一样)

进入/ws/vllm-fork/pd_xpyd目录，修改更新RECIPE_CACHE目录
修改dp_p_env.sh， 将
```bash
export PT_HPU_RECIPE_CACHE_CONFIG=/ws/cache-pd-p,false,16384
```
更新为
```bash
export PT_HPU_RECIPE_CACHE_CONFIG=/ws/cache-pd-1p2d-p,false,16384
```

修改dp_d_env.sh， 将
```bash
export PT_HPU_RECIPE_CACHE_CONFIG=/ws/cache-pd-d,false,16384
```
更新为
```bash
export PT_HPU_RECIPE_CACHE_CONFIG=/ws/cache-pd-1p2d-d,false,16384
```
进入/ws/vllm-fork/pd_xpyd，修改示例mooncake_3fg6.json完成设置

如果RDMA设备不可用，把协议设置为tcp

示例mooncake_3fg6.json中，decoding节点当前第一个Decode节点IP地址为192.168.1.106

metadata_server (etcd), master_server_address 和prefill节点是一样的

然后调整dp0_xp2d_start_decode.sh的最后一行，主要更新g13为本机机器名例如3fg6, "10.239.129.81"更新为本机以太网IP，例如示例中192.168.1.106
```
last line: source "$BASH_DIR"/dp_start_decode.sh g13 16 $TP_SIZE 0 "10.239.129.81" 
parameters are: machine, DP Size, TP Size, DP Index, DP Host IP
```

使用以下命令启动decoding服务器
```bash
source dp0_xp2d_start_decode.sh
```

### 准备decoding节点2
启动容器(步骤和prefill节点一样)

进入/ws/vllm-fork/pd_xpyd目录，修改更新RECIPE_CACHE目录
修改dp_p_env.sh， 将
```bash
export PT_HPU_RECIPE_CACHE_CONFIG=/ws/cache-pd-p,false,16384
```
更新为
```bash
export PT_HPU_RECIPE_CACHE_CONFIG=/ws/cache-pd-1p2d-p,false,16384
```

修改dp_d_env.sh， 将
```bash
export PT_HPU_RECIPE_CACHE_CONFIG=/ws/cache-pd-d,false,16384
```
更新为
```bash
export PT_HPU_RECIPE_CACHE_CONFIG=/ws/cache-pd-1p2d-d,false,16384
```
进入/ws/vllm-fork/pd_xpyd，复制mooncake_3fg6.json一份到mooncake_3fg7.json, 
如果RDMA设备不可用，把协议设置为tcp

示例mooncake_3fg7.json中，decoding节点的IP地址为当前第二个Decode节点IP为192.168.1.107

metadata_server (etcd), master_server_address 和prefill节点是一样的

然后调整dp1_xp2d_start_decode.sh的最后一行，主要更新g14为本机机器名例如3fg7, "10.239.129.81"更新为之前第一个Decode节点的以太网IP，例如示例中192.168.1.106 
```
last line: source "$BASH_DIR"/dp_start_decode.sh g14 16 $TP_SIZE 1 "10.239.129.81" 
parameters are: machine, DP Size, TP Size, DP Index, DP Host IP
```

使用以下命令启动decoding服务器
```bash
source dp1_xp2d_start_decode.sh
```

### 启动 PD proxy 服务器
代理服务器运行在prefill节点，编辑/ws/vllm-fork/pd_xpyd目录下的xpyd_start_proxy.sh，指定prefille server（变量：PREFILL_IPS）和decoding server（变量：DECODE_IPS）的IP地址。现在Prefill server是1个，decoding server为2个IP

使用以下命令启动PD proxy服务器
```bash
cd /ws/vllm-fork
bash pd_xpyd/xpyd_start_proxy.sh 1 2
```

### 验证
验证方法与1P1D一样

