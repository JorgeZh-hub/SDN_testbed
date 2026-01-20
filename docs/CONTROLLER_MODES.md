<!-- File: docs/CONTROLLER_MODES.md -->

# Modos del controlador (Ryu)

El controlador corre en el contenedor `controller_ryu` y se configura en:

- `containers/core/controller/generic_controller/ryu.conf` (parámetros/paths)
- `containers/core/controller/generic_controller/te_cfg.yml` (política TE/QoS + clasificación)

## Modos

- **Baseline (OSPF-like)**: Dijkstra con costo por capacidad
  (w(e)=K*c \cdot \frac{C*{ref}}{C_e})

- **TE_QOS_C (enlaces compartidos)**: reroute reactivo selectivo de flujos movibles ante congestión
  (w(e)=K*c \cdot \frac{C*{ref}}{C_e} + K_u \cdot \frac{L_e}{C_e})

- **TE_QOS_D (enlaces dedicados)**: dedicación por **grupos de colas** y locks TTL (no depende de costo global)

## Figuras (flujogramas)

Los PDFs listos para incluir en informes/tesis están en:

- `docs/figures/Dijsktra.pdf`
- `docs/figures/TE_QOS_C.pdf`
- `docs/figures/TE_QOS_D.pdf`
- `docs/figures/pipeline_scenario_runner.pdf`
