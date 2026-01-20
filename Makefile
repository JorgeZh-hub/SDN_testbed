SHELL := /bin/bash

# --- Ajustables por el usuario ---
TOP ?= src/topology/topology_conf1_v2.yml
CAP ?= src/experiments/capture_conf.yml
DUR ?= 120

OUT ?= results/tmp
CTRL_NAME ?= ryu-ctrl
SIM_NAME ?= containernet-sim
MN_PREFIX ?= mn.
PCAPS ?= $(OUT)/pcaps
REPORTS ?= $(OUT)/reports

# Si quieres usar docker sin sudo, cambia a: DOCKER := docker
DOCKER := sudo docker

.PHONY: help build-core build-services build run stop clean venv scenarios

help:
	@echo "Targets:"
	@echo "  help                  Show this help"
	@echo "  make deps             Install dependencies"
	@echo "  make build            Build core + services"
	@echo "  make run              Run single experiment (run_experiment.sh)"
	@echo "  make scenarios         Run scenario_runner batch"
	@echo "  make stop             Stop/remove ONLY testbed containers"
	@echo "  make clean            stop + cleanup OVS/Mininet (via container)"
	@echo ""
	@echo "Vars:"
	@echo "  TOP=..., CAP=..., DUR=120, OUT=..., CTRL_NAME=..., SIM_NAME=..., MN_PREFIX=mn."


deps:
	sudo apt update
	sudo apt install -y openvswitch-switch iproute2 tcpdump tshark python3 python3-venv python3-pip

build-core:
	$(DOCKER) build -t containernet_docker containers/core/containernet
	$(DOCKER) build -t controller_ryu       containers/core/controller

build-services:
	$(DOCKER) build -t mqtt_broker          containers/services/mqtt_broker
	$(DOCKER) build -t amqp_broker          containers/services/amqp_broker
	$(DOCKER) build -t iot_container        containers/services/iot_container
	$(DOCKER) build -t ftp_server           containers/services/ftp_server
	$(DOCKER) build -t http_server_vod      containers/services/http_server_vod
	$(DOCKER) build -t rtp_server_vod       containers/services/rtp_server_vod
	$(DOCKER) build -t rtp_emitter_vod      containers/services/rtp_emitter_vod
	$(DOCKER) build -t rtp_receiver_vod     containers/services/rtp_receiver_vod
	$(DOCKER) build -t rtp_container_voip   containers/services/rtp_container_voip

build: build-core build-services

run:
	@mkdir -p $(OUT)/pcaps $(OUT)/metrics $(OUT)/logs
	TOPOLOGY_FILE=$(TOP) CAPTURE_CONF=$(CAP) \
	sudo -E ./scripts/run_experiment.sh \
	  -d $(DUR) \
	  --capture-dir $(OUT)/pcaps \
	  --metrics-dir $(OUT)/metrics \
	  --log-dir $(OUT)/logs \
	  --capture-conf $(CAP)

stop:
	./scripts/stop_containers.sh --controller $(CTRL_NAME) --sim $(SIM_NAME) --prefix "$(MN_PREFIX)"

clean:
	./scripts/stop_containers.sh --controller $(CTRL_NAME) --sim $(SIM_NAME) --prefix "$(MN_PREFIX)" --mn-clean

venv:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install pandas pyyaml

scenarios: venv
	. .venv/bin/activate && python3 src/experiments/scenario_runner.py \
	  --base-topology $(TOP) \
	  --scenarios-csv src/experiments/escenarios.csv \
	  --capture-conf $(CAP) \
	  --duration $(DUR) \
	  --repetitions 5 \
	  --output-root results/runs


analyze: venv
	@mkdir -p $(REPORTS)
	. .venv/bin/activate && python3 src/analyzer/flow_resolver.py \
	  --flows $(CAP) \
	  --pcaps $(PCAPS) \
	  --output-dir $(REPORTS)
