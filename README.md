cd C:\Users\juanv\Downloads\imagenet-tiny
python -m venv .venv
.venv\Scripts\activate.bat

Linux
cd ~/imagenet-tiny
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt


macOS
cd ~/imagenet-tiny
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

2. Instalar dependencias

Con el entorno virtual activado:

Windows / macOS / Linux
python -m pip install --upgrade pip
python -m pip install -r requirements.txt




En otra terminal:

python worker.py --server-host 192.168.1.8 --server-port 5000 --data-dir "C:\Users\juanv\Downloads\imagenet-tiny\tiny-imagenet-200" --name worker1


Ejecutar worker Mac
python worker.py --server-host 192.168.1.8 --server-port 5000 --data-dir "$HOME/imagenet-tiny/tiny-imagenet-200" --name worker_mac_1

Ejecutar worker Linux
python worker.py --server-host 192.168.1.8 --server-port 5000 --data-dir "$HOME/imagenet-tiny/tiny-imagenet-200" --name worker_linux_1


python server.py --host 0.0.0.0 --port 5000 --data-dir "C:\Users\juanv\Downloads\imagenet-tiny\tiny-imagenet-200" --num-workers 2 --rounds 100 --local-epochs 1 --arch resnet18 --pretrained --optimizer adamw --lr 0.001 --lr-step-size 20 --lr-gamma 0.5 --weight-decay 0.0001 --train-batch-size 128 --val-batch-size 256 --loader-workers 2 --image-size 128 --max-val-samples 0 --save-model --save-best-model

python worker.py --server-host 127.0.0.1 --server-port 5000 --data-dir "C:\Users\juanv\Downloads\imagenet-tiny\tiny-imagenet-200" --name worker_windows

python3 worker.py --server-host 192.168.1.8 --server-port 5000 --data-dir "/Users/TU_USUARIO/ruta/tiny-imagenet-200" --name worker_mac

source .venv/bin/activate
python3 worker.py --server-host 192.168.1.8 --server-port 5000 --data-dir "/Users/TU_USUARIO/ruta/tiny-imagenet-200" --name worker_mac

recomendada
python server.py --host 0.0.0.0 --port 5000 --data-dir "C:\Users\juanv\Downloads\imagenet-tiny\tiny-imagenet-200" --num-workers 1 --rounds 15 --local-epochs 1 --arch resnet18 --pretrained --freeze-backbone --optimizer adamw --lr 0.001 --lr-step-size 10 --lr-gamma 0.5 --weight-decay 0.0001 --train-batch-size 128 --val-batch-size 256 --loader-workers 2 --image-size 128 --max-val-samples 2000 --save-model --save-best-model