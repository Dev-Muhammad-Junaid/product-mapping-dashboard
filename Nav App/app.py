from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import json
import os
from datetime import datetime
import uuid

app = Flask(__name__)
app.secret_key = 'nav-app-secret-key-change-in-production'
CORS(app)

# Data storage files
DATA_DIR = 'data'
DESTINATIONS_FILE = os.path.join(DATA_DIR, 'destinations.json')
LISTS_FILE = os.path.join(DATA_DIR, 'lists.json')
ROUTES_FILE = os.path.join(DATA_DIR, 'routes.json')

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize data files if they don't exist
def init_data_files():
    if not os.path.exists(DESTINATIONS_FILE):
        with open(DESTINATIONS_FILE, 'w') as f:
            json.dump([], f)
    if not os.path.exists(LISTS_FILE):
        with open(LISTS_FILE, 'w') as f:
            json.dump([], f)
    if not os.path.exists(ROUTES_FILE):
        with open(ROUTES_FILE, 'w') as f:
            json.dump([], f)

init_data_files()

# Helper functions
def load_json(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except:
        return []

def save_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/destinations', methods=['GET', 'POST'])
def destinations():
    if request.method == 'GET':
        destinations = load_json(DESTINATIONS_FILE)
        return jsonify(destinations)
    
    elif request.method == 'POST':
        data = request.json
        destinations = load_json(DESTINATIONS_FILE)
        
        new_destination = {
            'id': str(uuid.uuid4()),
            'name': data.get('name'),
            'emoji': data.get('emoji', '📍'),
            'color': data.get('color', '#4CAF50'),
            'x': data.get('x', 50),
            'y': data.get('y', 50),
            'category': data.get('category', 'general'),
            'created_at': datetime.now().isoformat()
        }
        
        destinations.append(new_destination)
        save_json(DESTINATIONS_FILE, destinations)
        
        return jsonify(new_destination), 201

@app.route('/api/destinations/<destination_id>', methods=['PUT', 'DELETE'])
def destination_detail(destination_id):
    destinations = load_json(DESTINATIONS_FILE)
    
    if request.method == 'PUT':
        data = request.json
        for dest in destinations:
            if dest['id'] == destination_id:
                dest.update({
                    'name': data.get('name', dest['name']),
                    'emoji': data.get('emoji', dest['emoji']),
                    'color': data.get('color', dest['color']),
                    'x': data.get('x', dest['x']),
                    'y': data.get('y', dest['y']),
                    'category': data.get('category', dest['category'])
                })
                save_json(DESTINATIONS_FILE, destinations)
                return jsonify(dest)
        return jsonify({'error': 'Destination not found'}), 404
    
    elif request.method == 'DELETE':
        destinations = [d for d in destinations if d['id'] != destination_id]
        save_json(DESTINATIONS_FILE, destinations)
        return jsonify({'message': 'Destination deleted'}), 200

@app.route('/api/routes', methods=['GET', 'POST'])
def routes():
    if request.method == 'GET':
        routes = load_json(ROUTES_FILE)
        return jsonify(routes)
    
    elif request.method == 'POST':
        data = request.json
        routes = load_json(ROUTES_FILE)
        
        new_route = {
            'id': str(uuid.uuid4()),
            'name': data.get('name', 'My Route'),
            'destinations': data.get('destinations', []),
            'color': data.get('color', '#2196F3'),
            'created_at': datetime.now().isoformat()
        }
        
        routes.append(new_route)
        save_json(ROUTES_FILE, routes)
        
        return jsonify(new_route), 201

@app.route('/api/routes/<route_id>', methods=['DELETE'])
def route_detail(route_id):
    routes = load_json(ROUTES_FILE)
    routes = [r for r in routes if r['id'] != route_id]
    save_json(ROUTES_FILE, routes)
    return jsonify({'message': 'Route deleted'}), 200

@app.route('/api/lists', methods=['GET', 'POST'])
def lists():
    if request.method == 'GET':
        lists = load_json(LISTS_FILE)
        return jsonify(lists)
    
    elif request.method == 'POST':
        data = request.json
        lists = load_json(LISTS_FILE)
        
        new_list = {
            'id': str(uuid.uuid4()),
            'name': data.get('name'),
            'description': data.get('description', ''),
            'category': data.get('category', 'general'),
            'destinations': data.get('destinations', []),
            'shared': data.get('shared', False),
            'creator': data.get('creator', 'Anonymous'),
            'created_at': datetime.now().isoformat()
        }
        
        lists.append(new_list)
        save_json(LISTS_FILE, lists)
        
        return jsonify(new_list), 201

@app.route('/api/lists/<list_id>', methods=['GET', 'PUT', 'DELETE'])
def list_detail(list_id):
    lists = load_json(LISTS_FILE)
    
    if request.method == 'GET':
        for lst in lists:
            if lst['id'] == list_id:
                return jsonify(lst)
        return jsonify({'error': 'List not found'}), 404
    
    elif request.method == 'PUT':
        data = request.json
        for lst in lists:
            if lst['id'] == list_id:
                lst.update({
                    'name': data.get('name', lst['name']),
                    'description': data.get('description', lst['description']),
                    'category': data.get('category', lst['category']),
                    'destinations': data.get('destinations', lst['destinations']),
                    'shared': data.get('shared', lst['shared'])
                })
                save_json(LISTS_FILE, lists)
                return jsonify(lst)
        return jsonify({'error': 'List not found'}), 404
    
    elif request.method == 'DELETE':
        lists = [l for l in lists if l['id'] != list_id]
        save_json(LISTS_FILE, lists)
        return jsonify({'message': 'List deleted'}), 200

@app.route('/api/planner', methods=['GET', 'POST'])
def planner():
    """Day planner endpoint"""
    PLANNER_FILE = os.path.join(DATA_DIR, 'planner.json')
    
    if not os.path.exists(PLANNER_FILE):
        with open(PLANNER_FILE, 'w') as f:
            json.dump([], f)
    
    if request.method == 'GET':
        planner = load_json(PLANNER_FILE)
        return jsonify(planner)
    
    elif request.method == 'POST':
        data = request.json
        save_json(PLANNER_FILE, data)
        return jsonify(data), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
