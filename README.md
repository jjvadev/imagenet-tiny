# Tiny ImageNet Distributed Training

Proyecto para entrenar **Tiny ImageNet-200** de forma distribuida con arquitectura **server-worker** usando sockets en Python y entrenamiento local con **gradiente descendente**.

La idea es simple:

- El **server** mantiene el modelo global.
- Cada **worker** recibe el modelo, entrena localmente con su particion del dataset usando **SGD** o **Adam**.
- El **server** agrega los pesos de todos los workers mediante un promedio ponderado por cantidad de muestras.
- Al finalizar cada round, el server evalua en validacion y guarda metricas claras.

## 1. Lo que mejora esta version

Esta version mejora una base inicial que ya tenia:

- generacion de notebook de analisis fileciteturn0file0
- utilidades de sockets para enviar y recibir mensajes serializados fileciteturn0file1
- construccion de modelos con `resnet18` y `mobilenet_v3_small` fileciteturn0file2

Ahora el proyecto incluye:

- `server.py` completo
- `worker.py` completo
- carga robusta de Tiny ImageNet
- soporte para `train`, `val` y lectura del `test`
- metrica de `loss`, `top1`, `top5`, `tiempo por round`, `tiempo total`, `throughput`
- guardado de `history.csv`, `summary.json`, `best_model.pt`, `last_model.pt`
- notebook `analysis.ipynb` generado automaticamente
- soporte para macOS, Windows y CPU/GPU/MPS

## 2. Estructura del proyecto

```text
 tiny_imagenet_distributed/
 |-- analysis_notebook.py
 |-- connection.py
 |-- data.py
 |-- models.py
 |-- training.py
 |-- server.py
 |-- worker.py
 |-- requirements.txt
 |-- README.md
 |-- scripts/
```

## 3. Algoritmo de entrenamiento

### Opcion por defecto: SGD

El entrenamiento local de cada worker usa **mini-batch gradient descent** con la variante **SGD con momentum**.

Formula conceptual:

```text
w(t+1) = w(t) - lr * gradiente
```

Con momentum:

```text
v(t+1) = momentum * v(t) - lr * gradiente
w(t+1) = w(t) + v(t+1)
```

### En distribuido

1. El server envia los pesos globales.
2. Cada worker entrena en su shard local.
3. Cada worker devuelve pesos actualizados y metricas.
4. El server calcula un promedio ponderado por numero de muestras.
5. Se evalua el nuevo modelo global en validacion.

Esto es una forma simple y efectiva de entrenamiento distribuido sincronico estilo **parameter server / FedAvg**.

## 4. Modelos soportados

- `small_cnn`
- `resnet18`
- `mobilenet_v3_small`

### Recomendacion practica

Para Tiny ImageNet, **si es viable y recomendable usar redes convolucionales**.

- **`small_cnn`**: util para pruebas rapidas, debugging o equipos limitados.
- **`resnet18`**: mejor punto de equilibrio entre calidad y costo.
- **`mobilenet_v3_small`**: buena opcion si quieres menor consumo de memoria o correr en equipos modestos.

### Mi recomendacion

- Si buscas mejores resultados: `resnet18 --pretrained`
- Si buscas velocidad: `mobilenet_v3_small --pretrained`
- Si buscas simplicidad academica para explicar gradiente descendente: `small_cnn`

## 5. Dataset Tiny ImageNet

Se asume la estructura clasica:

```text
tiny-imagenet-200/
|-- train/
|-- val/
|-- test/
```

### Importante sobre `test/`

En Tiny ImageNet original, muchas veces `test/` **no tiene labels publicos**.
Eso significa:

- puedes usar `val/` para medir accuracy real
- puedes usar `test/` para inferencia
- no siempre puedes calcular accuracy en `test/`

## 6. Instalacion

## macOS

### Crear entorno virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Windows CMD

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 7. Como ejecutar

### Paso 1: iniciar el server

Ejemplo para tu estructura en macOS:

```bash
python server.py \
  --host 0.0.0.0 \
  --port 5001 \
  --num-workers 2 \
  --rounds 5 \
  --local-epochs 1 \
  --arch resnet18 \
  --pretrained \
  --lr 0.01 \
  --optimizer sgd \
  --momentum 0.9 \
  --weight-decay 1e-4 \
  --batch-size 64 \
  --eval-batch-size 128 \
  --image-size 64 \
  --augmentation \
  --val-dir "/Users/zarko1989/Documents/tiny-imagenet-200/val" \
  --results-dir results
```

### Paso 2: iniciar los workers

#### Worker 0 en macOS

```bash
python worker.py \
  --server-host 127.0.0.1 \
  --server-port 5001 \
  --worker-id 0 \
  --num-workers 2 \
  --train-dir "/Users/zarko1989/Documents/tiny-imagenet-200/train" \
  --batch-size 64 \
  --image-size 64 \
  --augmentation
```

#### Worker 1 en macOS

```bash
python worker.py \
  --server-host 127.0.0.1 \
  --server-port 5001 \
  --worker-id 1 \
  --num-workers 2 \
  --train-dir "/Users/zarko1989/Documents/tiny-imagenet-200/train" \
  --batch-size 64 \
  --image-size 64 \
  --augmentation
```

## 8. Comandos equivalentes en Windows

### Server

```powershell
python server.py `
  --host 0.0.0.0 `
  --port 5001 `
  --num-workers 2 `
  --rounds 5 `
  --local-epochs 1 `
  --arch resnet18 `
  --pretrained `
  --lr 0.01 `
  --optimizer sgd `
  --momentum 0.9 `
  --weight-decay 1e-4 `
  --batch-size 64 `
  --eval-batch-size 128 `
  --image-size 64 `
  --augmentation `
  --val-dir "C:\ruta\tiny-imagenet-200\val" `
  --results-dir results
```

### Worker 0

```powershell
python worker.py `
  --server-host 127.0.0.1 `
  --server-port 5001 `
  --worker-id 0 `
  --num-workers 2 `
  --train-dir "C:\ruta\tiny-imagenet-200\train" `
  --batch-size 64 `
  --image-size 64 `
  --augmentation
```

### Worker 1

```powershell
python worker.py `
  --server-host 127.0.0.1 `
  --server-port 5001 `
  --worker-id 1 `
  --num-workers 2 `
  --train-dir "C:\ruta\tiny-imagenet-200\train" `
  --batch-size 64 `
  --image-size 64 `
  --augmentation
```

## 9. Que resultados genera

El server guarda dentro de `results/`:

- `history.csv`
- `summary.json`
- `best_model.pt`
- `last_model.pt`
- `analysis.ipynb`

### Metricas por round

- `train_loss`
- `train_top1`
- `train_top5`
- `val_loss`
- `val_top1`
- `val_top5`
- `round_time_sec`
- `total_time_sec`
- `throughput_samples_per_sec`

## 10. Ejemplo de salida en consola

```text
[Server] Round 1/5
  Worker 0: loss=4.9210, top1=0.1020, top5=0.2510, samples=50000, time=95.31s
  Worker 1: loss=4.8472, top1=0.1105, top5=0.2662, samples=50000, time=96.04s
[Server] Round 1 | train_loss=4.8841 | train_top1=0.1062 | train_top5=0.2586 | val_loss=4.5207 | val_top1=0.1554 | val_top5=0.3621 | time=103.45s | throughput=967.52 samples/s
```

## 11. Recomendaciones para obtener resultados mas claros

- Usa `resnet18 --pretrained` para mejorar accuracy.
- Mantiene `image_size=64` para respetar Tiny ImageNet.
- Empieza con `rounds=5` y luego sube a `20` o `30`.
- Usa `local-epochs=1` o `2` para evitar divergencia entre workers.
- Si el equipo tiene poca RAM, baja `batch-size` a `32`.
- Si entrenas en Apple Silicon, usa `--device mps`.
- Si entrenas con NVIDIA, usa `--device cuda`.

## 12. Ejemplos recomendados

### Entrenamiento rapido para validacion tecnica

```bash
python server.py --num-workers 2 --rounds 3 --local-epochs 1 --arch small_cnn --lr 0.01 --optimizer sgd --val-dir "/Users/zarko1989/Documents/tiny-imagenet-200/val"
```

### Entrenamiento rapido para validacion tecnica

```bash windows
python server.py --num-workers 2 --rounds 3 --local-epochs 1 --arch small_cnn --lr 0.01 --optimizer sgd --val-dir "\juanv\Downloads\imagenet-tiny\val"
```

### Mejor balance calidad/velocidad

```bash
python server.py --num-workers 2 --rounds 10 --local-epochs 1 --arch resnet18 --pretrained --lr 0.005 --optimizer sgd --momentum 0.9 --val-dir "/Users/zarko1989/Documents/tiny-imagenet-200/val"
```

### Alternativa ligera

```bash
python server.py --num-workers 2 --rounds 10 --local-epochs 1 --arch mobilenet_v3_small --pretrained --lr 0.005 --optimizer adam --val-dir "/Users/zarko1989/Documents/tiny-imagenet-200/val"
```

## 13. Limitaciones actuales

- No hay scheduler de learning rate.
- No hay reanudacion de checkpoints intermedios.
- No hay DDP nativo de PyTorch; aqui se usa arquitectura por sockets.
- `test/` puede no tener etiquetas.

## 14. Mejoras futuras recomendadas

- agregar cosine scheduler
- guardar confusion matrix
- agregar early stopping
- soportar reanudacion desde checkpoint
- agregar inferencia sobre `test/`
- exportar resultados a PDF o HTML

## 15. Conclusion

Si, **manejar redes convolucionales en este caso es totalmente viable y recomendable**.
Para Tiny ImageNet, una CNN o una red pretrained tipo ResNet suele ser mucho mejor opcion que un modelo denso simple.

Si quieres una configuracion final recomendada para empezar hoy mismo:

- `arch=resnet18`
- `pretrained=True`
- `optimizer=sgd`
- `lr=0.005` o `0.01`
- `momentum=0.9`
- `rounds=10`
- `local_epochs=1`
- `batch_size=64`

Cuba  ip casa
python worker.py \
  --server-host 192.168.1.2 \
  --server-port 5001 \
  --worker-id 0 \
  --num-workers 2 \
  --train-dir "/Users/zarko1989/Documents/tiny-imagenet-200/train" \
  --batch-size 64 \
  --image-size 64 \
  --augmentation

  www