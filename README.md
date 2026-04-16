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
