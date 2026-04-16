source .venv/bin/activate

python worker.py \
  --server-host 192.168.1.8 \
  --server-port 5000 \
  --data-dir "$HOME/imagenet-worker/tiny-imagenet-200" \
  --name macworker1


LINUX
cd ~/imagenet-worker
source .venv/bin/activate
python worker.py \
  --server-host 192.168.1.8 \
  --server-port 5000 \
  --data-dir "$HOME/imagenet-worker/tiny-imagenet-200" \
  --name linuxworker1


  para actualizar al ultimo commit