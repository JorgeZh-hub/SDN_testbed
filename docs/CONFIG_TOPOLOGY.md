<!-- File: docs/CONFIG_TOPOLOGY.md -->

# Configurar una topología (`topology_conf*.yml`)

> La topología YAML es la entrada principal del testbed. Se procesa con `src/topology/topology_test.py`.

## Secciones del YAML

- `controller:` imagen/comando/volúmenes del contenedor **Ryu**.
- `qos:` (opcional) colas en OVS (HTB) con `queues` y `default_queue_id`.
- `hosts:` contenedores (clientes/servidores) con IPs y comandos.
- `n_switches:` número de switches (y nombres si aplica en tu versión).
- `links:` enlaces `"nodo-if, nodo-if, cap_mbps"`.

## IP plan (RFC 5737)

Para documentación/laboratorios se usan:

- `198.51.100.0/24`
- `203.0.113.0/24`

## QoS HTB (colas)

En `qos:` define:

- `enabled: true`
- `type: "linux-htb"`
- `default_queue_id: 0`
- `queues: [{id:0,...}, {id:1,...}, ...]`

Cada `id` se usa desde el controlador con `SetQueue(queue_id)`.

## Flujos y carga (`n_proc`)

En cada host puedes definir `flows:` (tráfico).
`n_proc` es el número de instancias del flujo (lo ajusta `scenario_runner.py` desde `escenarios.csv`).

## Ejemplos visuales (figuras)

- `docs/figures/topology_examples/topology_conf1_v1.png`
- `docs/figures/topology_examples/topology_conf1_v2.png`
- `docs/figures/topology_examples/topology_conf1_v3.png`
