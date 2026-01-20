# SDN Testbed (Containernet + Ryu OpenFlow 1.3) – TE + QoS for Heterogeneous Traffic

Este repositorio contiene el banco de pruebas usado en mi tesis para emular una WAN SDN con **Containernet/OVS** y un controlador **Ryu (OpenFlow 1.3)**, evaluando tres modos de encaminamiento:

* **Baseline (OSPF-like)**: Dijkstra con costo basado en capacidad del enlace.
* **TE_QOS_C (enlaces compartidos)**: reroute reactivo selectivo de flujos movibles cuando hay congestión.
* **TE_QOS_D (enlaces dedicados)**: dedicación reactiva de enlaces por **grupos de colas** con locks TTL.

Incluye tráfico IoT (MQTT/CoAP/HTTP/AMQP), multimedia (RTP/RTSP/HLS), FTP y ruido UDP, con generación **tipo ráfagas (CTMC N-Burst)**.

---

## Requisitos (host)

| Componente    |      Versión recomendada | Notas                         |
| ------------- | -----------------------: | ----------------------------- |
| Ubuntu (host) |            22.04 / 24.04 | Probado en estas versiones    |
| Docker Engine |                     28.x | Requerido                     |
| Open vSwitch  |                      3.x | En el host (para switches)    |
| tcpdump       |                    4.99+ | Captura PCAP en interfaces    |
| tshark        |                      4.x | Análisis automático de PCAPs  |
| Python (host) | 3.10+ (recomendado 3.12) | Post-proceso (pandas, pyyaml) |

> **IPs:** las IPs del laboratorio usan los bloques de documentación del **RFC 5737** (198.51.100.0/24 y 203.0.113.0/24).

---

## Estructura del repo (qué hace cada carpeta)

* `containers/`

  * `core/containernet/` → Dockerfile de **containernet_docker**
  * `core/controller/` → Dockerfile de **controller_ryu** (Ryu + controlador)
  * `services/` → imágenes de servicios: MQTT, AMQP, IoT, VoD, VoIP, FTP
* `src/`

  * `topology/` → levanta topología desde YAML + QoS HTB + tráfico ráfagas (CTMC)
  * `experiments/` → escenarios masivos (`scenario_runner.py`, `escenarios.csv`, `capture_conf.yml`)
  * `analyzer/` → análisis de PCAPs con tshark (`flow_resolver.py`)
* `scripts/`

  * `run_experiment.sh` → orquesta un experimento (controller + containernet + capturas)
  * `capture_links.sh` / `capture_containers_metrics.sh` → tcpdump + docker stats
* `results/pipeline/` → scripts para consolidar resultados y generar tablas/reportes
* `data/` → insumos locales (ej. VoIP/FTP) para generar tráfico real.

---

## Construcción de imágenes (una vez)

```bash
# Core
docker build -t containernet_docker:latest containers/core/containernet
docker build -t controller_ryu:latest containers/core/controller

# Servicios
docker build -t mqtt_broker:latest containers/services/mqtt_broker
docker build -t amqp_broker:latest containers/services/amqp_broker
docker build -t iot_container:latest containers/services/iot_container
docker build -t ftp_server:latest containers/services/ftp_server
docker build -t http_server_vod:latest containers/services/http_server_vod
docker build -t rtp_server_vod:latest containers/services/rtp_server_vod
docker build -t rtp_emitter_vod:latest containers/services/rtp_emitter_vod
docker build -t rtp_receiver_vod:latest containers/services/rtp_receiver_vod
docker build -t rtp_container_voip:latest containers/services/rtp_container_voip
```

---

## Quickstart 1: correr UNA topología (ejemplo manual)

```bash
export HOST_PROJECT_ROOT="$(pwd)"
export TOPOLOGY_FILE="src/topology/topology_conf1_v3.yml"

sudo -E ./scripts/run_experiment.sh \
  --duration 120 \
  --capture-dir results/tmp/pcaps \
  --metrics-dir results/tmp/containers \
  --log-dir results/tmp/logs \
  --capture-conf src/experiments/capture_conf.yml
```

---

## Quickstart 2: correr escenarios masivos (recomendado)

`escenarios.csv` define el `n_proc` (número de procesos/instancias) por flujo.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install pandas pyyaml

python3 src/experiments/scenario_runner.py \
  --base-topology src/topology/topology_conf1_v3.yml \
  --scenarios-csv src/experiments/escenarios.csv \
  --capture-conf src/experiments/capture_conf.yml \
  --duration 120 \
  --repetitions 5 \
  --output-root results/runs
```

Cada escenario genera:

* `results/runs/<escenario>/topology_conf.yml`
* `rep_*/pcaps/` + `rep_*/reports/` + `rep_*/logs/`

---

## Configuración del controlador (IMPORTANTE)

El controlador se configura en **dos archivos**:

1. `containers/core/controller/generic_controller/ryu.conf`
   Define: `top_config`, `te_config`, periodos, umbrales, modo TE, Kc/Ku, etc.

2. `containers/core/controller/generic_controller/te_cfg.yml`
   Define: colas/clases, reglas de clasificación (MQTT/CoAP/RTP/HTTP…), colas protegidas y (en TE_QOS_D) `queue_groups` + locks TTL.

> Nota: `ryu.conf` usa rutas **dentro del contenedor** (ej. `/app/exp/topology_conf.yml` y `/app/generic_controller/te_cfg.yml`). Asegúrate que coincidan con los volúmenes definidos en `topology_conf.yml`.

---

## Documentación extendida

* `docs/INSTALL.md` → instalación host + validaciones
* `docs/CONFIG_TOPOLOGY.md` → cómo armar `topology_conf.yml` (controller/hosts/links/qos)
* `docs/EXPERIMENTS.md` → `run_experiment.sh` vs `scenario_runner.py`, estructura de resultados
* `docs/ANALYSIS.md` → `capture_conf.yml` + `flow_resolver.py`
* `docs/CONTROLLER_MODES.md` → OSPF vs TE_QOS_C vs TE_QOS_D (con flujogramas)

---

## License / Citation

Si usas este repo en investigación, por favor cita la tesis (ver `CITATION.cff`).
