<!-- File: docs/ANALYSIS.md -->

# Análisis (PCAPs y métricas)

El análisis de flujos usa `tshark` mediante:

- `src/analyzer/flow_resolver.py`

## 1) Configuración: `capture_conf.yml`

- `flows:` define qué flujos medir (enlaces lógicos hostA-hostB + protocolo/puertos).
- `containers:` lista de contenedores a monitorear con `docker stats`.

## 2) Qué calcula `flow_resolver.py`

Produce CSV por flujo con:

- **delay** (retardo)
- **jitter** (variación del retardo)
- **throughput** (tasa efectiva)
- **loss** (pérdidas estimadas)

## 3) Ejecución recomendada (si agregaste target)

```bash
make analyze
```

## 4) Ejecución manual

```bash
python3 src/analyzer/flow_resolver.py \
  --flows src/experiments/capture_conf.yml \
  --pcaps results/tmp/pcaps \
  --output-dir results/tmp/reports/flows
```

Ayuda:

```bash
python3 src/analyzer/flow_resolver.py -h
```
