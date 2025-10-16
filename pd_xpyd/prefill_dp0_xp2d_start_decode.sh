BASH_DIR=$(dirname "${BASH_SOURCE[0]}")

source "$BASH_DIR"/start_etcd_mooncake_master.sh

if [ -z "$1" ]; then
    echo "please input the tp size"
    echo "run with default mode n=1"
    TP_SIZE=1
else
    TP_SIZE=$1
fi

source "$BASH_DIR"/dp_start_prefill.sh g10 16 $TP_SIZE 0 "10.239.129.9"
