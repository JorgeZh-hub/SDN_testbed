from flask import Flask, request, jsonify
from collections import deque, defaultdict
import threading
import copy

app = Flask(__name__)

# CONFIGURACIÓN
# In-memory FIFO history
history = deque(maxlen=500)
# Track delivered measurement IDs per client IP
sent_per_ip = defaultdict(set)

# MUTEX LOCK (Esto es vital para la estabilidad bajo tráfico)
data_lock = threading.Lock()

@app.route('/datos', methods=['POST'])
def recibir_datos():
    data = request.json
    if not data or not isinstance(data, dict):
        return "Invalid format", 400

    # Bloqueamos para escribir de forma segura
    with data_lock:
        history.append(data)
    
    return "OK", 200

@app.route('/datos/<tipo_y_id>', methods=['GET'])
def obtener_datos(tipo_y_id):
    ip = request.remote_addr
    
    if '_' not in tipo_y_id:
        return "Invalid format. Use /datos/tipo_id", 400

    tipo, identificador = tipo_y_id.split('_', 1)
    clave = f"id_{tipo}"

    nuevos = []
    
    # Bloqueamos SOLO para leer la historia y copiar lo necesario
    # Esto evita que el loop falle si entra un POST al mismo tiempo
    with data_lock:
        # Hacemos una copia rápida o iteramos protegidos
        snapshot = list(history) 
        # Accedemos al set del usuario dentro del lock por seguridad
        already_sent = sent_per_ip[ip]
    
    # Procesamos fuera del lock para no frenar el servidor
    # (Ya tenemos el snapshot de datos)
    for dato in snapshot:
        if dato.get(clave) == identificador:
            id_medida = dato.get("id_medida")
            if id_medida and id_medida not in already_sent:
                nuevos.append(dato)
                # Ojo: aquí modificamos el estado del usuario, necesitamos proteger esto
                # O simplemente asumimos un riesgo menor, pero lo ideal es:
                with data_lock:
                    sent_per_ip[ip].add(id_medida)

    if nuevos:
        return jsonify(nuevos), 200
    return f"No new data for {tipo}_{identificador}", 404

@app.route('/datos/todos', methods=['GET'])
def obtener_todos():
    with data_lock:
        # Convertimos a lista dentro del lock
        return jsonify(list(history)), 200

@app.route('/datos/limpiar', methods=['DELETE'])
def limpiar():
    with data_lock:
        history.clear()
        sent_per_ip.clear()
    return "History cleared", 200

# ELIMINA EL BLOQUE "if __name__ == '__main__': app.run..."
# No lo uses para tráfico real.