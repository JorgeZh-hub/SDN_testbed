#!/bin/bash

#🛠️ Paso a paso detallado
#✅ 1. Crear un bridge virtual en tu host para conectar a Containernet
sudo ip link add name br-virt type bridge
sudo ip link set br-virt up
#✅ 2. Crear un par de interfaces veth
sudo ip link add veth-host type veth peer name veth-cont
#✅ 3. Conectar un extremo (veth-host) al bridge
sudo ip link set veth-host master br-virt

# Asignar ip 
sudo ip addr add 10.10.0.1/24 dev br-virt
sudo ip link set br-virt up
