<!-- File: docs/EXPERIMENTS.md -->

# Ejecutar experimentos

Camino recomendado: **Makefile**.
Los scripts también aceptan `-h/--help` para ver opciones.

## 1) Una corrida (recomendado)

```bash
make run TOP=src/topology/topology_conf1_v2.yml DUR=120
```

Qué genera (típico):

- `results/tmp/pcaps/` (PCAPs por interfaz)
- `results/tmp/metrics/` (docker stats)
- `results/tmp/logs/` (logs)

## 2) Escenarios masivos (recomendado)

```bash
make scenarios TOP=src/topology/topology_conf1_v2.yml DUR=120
```

Salida típica:
`results/runs/<escenario>/rep_<k>/{pcaps,metrics,logs,reports}`

## 3) Avanzado: ejecutar scripts directo

Una corrida manual:

```bash
TOPOLOGY_FILE=src/topology/topology_conf1_v2.yml \
CAPTURE_CONF=src/experiments/capture_conf.yml \
sudo -E ./scripts/run_experiment.sh -d 120
```

Ayuda:

```bash
./scripts/run_experiment.sh -h
python3 src/experiments/scenario_runner.py -h
```
