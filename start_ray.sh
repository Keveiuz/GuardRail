export CUDA_VISIBLE_DEVICES=4,5,6,7
ray start --head \
  --port=6360 \
  --dashboard-port=8269 \
  --num-gpus=4 \
  --num-cpus=32 \
  --min-worker-port=14000 \
  --max-worker-port=14100