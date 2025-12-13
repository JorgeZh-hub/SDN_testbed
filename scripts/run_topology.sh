#!/bin/bash

# Virtual environment
source ~Documents/Geovanny/UIC/Python_UIC/bin/activate

sudo ip addr flush dev eno1        # Delete IP eno1
# Network

# Run topology
sudo PYTHONPATH=~/Documents/Geovanny/UIC/Red/containernet python3 topology.py

# Activate eno
#sudo ip addr add 203.0.113.101/24 dev eno2
sudo ip link set eno1 up