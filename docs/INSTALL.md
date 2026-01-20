<!-- File: docs/INSTALL.md -->

# Instalación (camino recomendado)

> Objetivo: dejar el host listo para correr el testbed usando **Makefile**.
> Nota: los scripts del repo soportan `-h/--help` (ej.: `./scripts/run_experiment.sh -h`).

## 1) Requisitos (host)

- Ubuntu 22.04 / 24.04
- Docker Engine, Open vSwitch, tcpdump, tshark, Python 3.10+ (recomendado 3.12)

## 2) Instalar dependencias

Recomendado (con Makefile):

```bash
make deps
```

Alternativa (manual):

```bash
sudo apt update
sudo apt install -y openvswitch-switch iproute2 tcpdump tshark python3 python3-venv python3-pip
```

## 3) Chequeos rápidos

```bash
docker version
ovs-vsctl --version
tshark -v
tcpdump --version
python3 --version
```

## 4) Permisos y por qué `sudo -E`

- `tcpdump` requiere privilegios para capturar interfaces virtuales de OVS/Containernet.
- El repo usa `sudo -E` para **preservar variables de entorno** (p. ej. `TOPOLOGY_FILE`, rutas de salida) al ejecutar scripts.

Opcional (tshark sin sudo):

```bash
sudo dpkg-reconfigure wireshark-common
sudo usermod -aG wireshark $USER
# relogin
```

## 5) Build de imágenes (una vez)

```bash
make build
```

Ver qué hace cada target:

```bash
make help
```
