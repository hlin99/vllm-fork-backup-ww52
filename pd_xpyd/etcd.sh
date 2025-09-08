#!/bin/bash
set -e

# ===========================
# 配置
# ===========================
MEM_DIR="/mnt/etcd_mem"
MEM_SIZE="100G"
ETCD_NAME="node1"
#IP="192.168.100.105"       # 本机内网IP
IP="10.239.129.24"
CLIENT_PORT="2379"
PEER_PORT="26680"
CLUSTER_TOKEN="etcd-cluster-1"
LOG_FILE="etcd_mem.log"

# ===========================
# 1️⃣ 创建并挂载内存目录
# ===========================
echo "挂载内存目录 $MEM_DIR，大小 $MEM_SIZE ..."
sudo mkdir -p $MEM_DIR
sudo mount -t tmpfs -o size=$MEM_SIZE tmpfs $MEM_DIR

# ===========================
# 2️⃣ 清理可能存在的环境变量
# ===========================
unset ETCD_DATA_DIR ETCD_NAME ETCD_LISTEN_CLIENT_URLS ETCD_ADVERTISE_CLIENT_URLS \
      ETCD_LISTEN_PEER_URLS ETCD_INITIAL_ADVERTISE_PEER_URLS ETCD_INITIAL_CLUSTER \
      ETCD_INITIAL_CLUSTER_STATE ETCD_INITIAL_CLUSTER_TOKEN

# ===========================
# 3️⃣ 启动 etcd（只用命令行参数，避免冲突）
# ===========================
echo "启动 etcd 服务 ..."
etcd \
  --name "$ETCD_NAME" \
  --data-dir "$MEM_DIR" \
  --listen-client-urls "http://0.0.0.0:$CLIENT_PORT" \
  --advertise-client-urls "http://$IP:$CLIENT_PORT" \
  --listen-peer-urls "http://$IP:$PEER_PORT" \
  --initial-advertise-peer-urls "http://$IP:$PEER_PORT" \
  --initial-cluster "$ETCD_NAME=http://$IP:$PEER_PORT" \
  --initial-cluster-state new \
  --initial-cluster-token "$CLUSTER_TOKEN" \
  >"$LOG_FILE" 2>&1 &

echo "etcd 启动完成，日志保存在 $LOG_FILE"
echo "客户端端口: $CLIENT_PORT, Peer端口: $PEER_PORT"

