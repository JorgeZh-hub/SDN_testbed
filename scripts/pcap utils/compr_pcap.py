from scapy.all import rdpcap, wrpcap

# Parámetros de entrada
input_pcap = "g711a_01.pcap"
output_pcap = "g711a.pcap"
factor_reduccion = 2  # Mantener 1 de cada 3 paquetes

# Cargar paquetes originales
packets = rdpcap(input_pcap)

# Submuestreo: conservar solo 1 de cada N paquetes
reduced_packets = [pkt for i, pkt in enumerate(packets) if i % factor_reduccion == 0]

# Guardar el nuevo PCAP con menor densidad
wrpcap(output_pcap, reduced_packets)
