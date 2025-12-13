from scapy.all import rdpcap, wrpcap

# Parámetros de entrada
input_pcap = "g711a.pcap"
output_pcap = "g711a_dense.pcap"
factor_aceleracion = 5  # Acelerar 5 veces => más paquetes por segundo

# Cargar paquetes originales
packets = rdpcap(input_pcap)

# Obtener el tiempo base
t0 = packets[0].time

# Aplicar compresión de tiempo
for pkt in packets:
    pkt.time = t0 + (pkt.time - t0) / factor_aceleracion

# Guardar PCAP modificado
wrpcap(output_pcap, packets)
