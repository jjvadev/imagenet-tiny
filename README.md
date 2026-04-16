source .venv/bin/activate

python worker.py \
  --server-host 192.168.1.8 \
  --server-port 5000 \
  --data-dir "$HOME/imagenet-worker/tiny-imagenet-200" \
  --name macworker1

windows
python server.py `
  --data-dir "C:\Users\juanv\Downloads\imagenet-tiny\tiny-imagenet-200" `
  --num-workers 2 `
  --rounds 50 `
  --local-epochs 1 `
  --arch small_cnn `
  --optimizer adam `
  --lr 0.001 `
  --train-batch-size 128 `
  --val-batch-size 256 `
  --loader-workers 2 `
  --max-val-samples 2000

python worker.py `
  --server-host 127.0.0.1 `
  --server-port 5000 `
  --data-dir "C:\Users\juanv\Downloads\imagenet-tiny\tiny-imagenet-200" `
  --name worker1


LINUX
cd ~/imagenet-worker
source .venv/bin/activate
python worker.py \
  --server-host 192.168.1.8 \
  --server-port 5000 \
  --data-dir "$HOME/imagenet-worker/tiny-imagenet-200" \
  --name linuxworker1


  para actualizar al ultimo commit