# SDN Testbed (Containernet + Ryu OpenFlow 1.3) – TE + QoS for Heterogeneous Traffic

Banco de pruebas usado en mi tesis para emular una WAN SDN con **Containernet/OVS** y un controlador **Ryu (OpenFlow 1.3)**, evaluando tres modos de encaminamiento:

- **Baseline (OSPF-like)**: Dijkstra con costo por capacidad del enlace.
- **TE_QOS_C (enlaces compartidos)**: reroute reactivo selectivo de flujos movibles ante congestión.
- **TE_QOS_D (enlaces dedicados)**: dedicación reactiva de enlaces por **grupos de colas** con locks TTL.

Incluye tráfico IoT (MQTT/CoAP/HTTP/AMQP), multimedia (RTP/RTSP/HLS), FTP y ruido UDP, con generación **tipo ráfagas (CTMC N-Burst)**.

> **IPs:** se usan rangos de documentación del **RFC 5737** (198.51.100.0/24 y 203.0.113.0/24).

---

## Requisitos (host)

Ubuntu 22.04/24.04, Docker Engine, Open vSwitch, tcpdump, tshark, Python 3.10+.

Detalles y validaciones: **[`docs/INSTALL.md`](docs/INSTALL.md)**

---

## Instalación desde cero (clone + quickstart)

```bash
git clone <URL_DE_TU_REPO_GITHUB>.git
cd testbed
```

Build de imágenes (una vez):

```bash
make build
```

Correr un experimento (una topología):

```bash
make run TOP=src/topology/topology_conf1_v2.yml DUR=120
```

Escenarios masivos (batch):

```bash
make scenarios TOP=src/topology/topology_conf1_v2.yml DUR=120
```

Parar y limpiar solo lo del testbed:

```bash
make stop
make clean
```

> Tip: la mayoría de scripts soportan `-h/--help` (ej.: `./scripts/run_experiment.sh -h`).

---

## Estructura del repo (resumen)

- `containers/` imágenes Docker (controlador, containernet, servicios)
- `src/topology/` carga topologías YAML + QoS HTB + tráfico ráfagas
- `src/experiments/` runner masivo (`scenario_runner.py`, `escenarios.csv`, `capture_conf.yml`)
- `src/analyzer/` análisis de PCAPs con tshark (`flow_resolver.py`)
- `scripts/` orquestación (controller + containernet + capturas)
- `docs/` documentación replicable + figuras

---

## Documentación

- [`docs/CONFIG_TOPOLOGY.md`](docs/CONFIG_TOPOLOGY.md) – YAML (controller/hosts/links/qos) + RFC 5737
- [`docs/CONTROLLER_MODES.md`](docs/CONTROLLER_MODES.md) – OSPF vs TE_QOS_C vs TE_QOS_D (+ figuras)
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) – `run_experiment.sh` vs `scenario_runner.py`
- [`docs/ANALYSIS.md`](docs/ANALYSIS.md) – `capture_conf.yml` + `flow_resolver.py`

---

## License / Citation

Si usas este repo en investigación, por favor cita la tesis (ver `CITATION.cff`).
