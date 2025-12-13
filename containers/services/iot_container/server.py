from flask import Flask, request, jsonify
from collections import deque, defaultdict

app = Flask(__name__)

# In-memory FIFO history
history = deque(maxlen=500)

# Track delivered measurement IDs per client IP
sent_per_ip = defaultdict(set)

@app.route('/datos', methods=['POST'])
def recibir_datos():
    data = request.json
    if not data or not isinstance(data, dict):
        return "Invalid format", 400

    history.append(data)
    return "OK", 200

@app.route('/datos/<tipo_y_id>', methods=['GET'])
def obtener_datos(tipo_y_id):
    ip = request.remote_addr
    already_sent = sent_per_ip[ip]

    if '_' not in tipo_y_id:
        return "Invalid format. Use /datos/tipo_id", 400

    tipo, identificador = tipo_y_id.split('_', 1)
    clave = f"id_{tipo}"

    nuevos = []
    for dato in history:
        if dato.get(clave) == identificador:
            id_medida = dato.get("id_medida")
            if id_medida and id_medida not in already_sent:
                nuevos.append(dato)
                already_sent.add(id_medida)

    if nuevos:
        return jsonify(nuevos), 200
    return f"No new data for {tipo}_{identificador}", 404

@app.route('/datos/todos', methods=['GET'])
def obtener_todos():
    return jsonify(list(history)), 200

@app.route('/datos/limpiar', methods=['DELETE'])
def limpiar():
    history.clear()
    sent_per_ip.clear()
    return "History cleared", 200

if __name__ == '__main__':
    print("Flask server listening on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000)
