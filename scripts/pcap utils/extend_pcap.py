from scapy.all import rdpcap, wrpcap, UDP
import copy

# Cargar el archivo original
packets = rdpcap("g711a_dense.pcap")
new_packets = []

# Cuántas veces repetir
repeticiones = 5
offset_tiempo = packets[-1].time - packets[0].time + 0.02  # Separación entre repeticiones

for i in range(repeticiones):
    for pkt in packets:
        p = copy.deepcopy(pkt)
        p.time += i * offset_tiempo
        new_packets.append(p)

# Guardar nuevo archivo
wrpcap("g711a_new.pcap", new_packets)
