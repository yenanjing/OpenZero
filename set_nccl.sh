export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_ALGO=ring
export NCCL_PROTO=ucp
export NCCL_SOCKET_IFNAME=bond1
export NCCL_MIN_NRINGS=1

master_addr="30.159.160.47"
export MASTER_ADDR=$master_addr
master_port=14545
export MASTER_PORT=$master_port
export NCCL_IB_GID_INDEX=3
node_rank=0
echo ${node_rank}
