from flask import Flask, request, jsonify
from flask_cors import CORS
import torch

from environment_predictor import RoomTemperaturePredictor, generate_room_data, train_and_evaluate
from optimization import optimize_hvac

app = Flask(__name__)
CORS(app)

# Global variables for the model
env_model = None
mae = 0.5
current_target = 24.0

def initialize_model():
    global env_model, mae
    print("Initializing System and Training Brain...")
    X_data, y_data = generate_room_data(2000)
    env_model = RoomTemperaturePredictor()
    
    # Train and evaluate model to get mae
    split_idx = int(len(X_data) * 0.8)
    X_test, y_test = X_data[split_idx:], y_data[split_idx:]
    
    env_model = train_and_evaluate(env_model, X_data, y_data, epochs=150)
    
    # Calculate MAE
    env_model.eval()
    with torch.no_grad():
        test_preds = env_model(X_test)
        mae = torch.mean(torch.abs(test_preds - y_test)).item()

# Initialize on startup
initialize_model()

@app.route('/api/optimize', methods=['POST'])
def optimize():
    global current_target
    data = request.json
    city = data.get('city', 'Delhi')
    target_temp = data.get('target_temp', current_target)
    current_target = target_temp
    
    # The frontend allows passing epochs, but optimization.py has it hardcoded for now.
    # We will use the optimize_hvac function directly.
    result = optimize_hvac(target_temp, city, env_model)
    
    if not result:
        return jsonify({"error": "Failed to fetch weather data for the given city."}), 400
        
    result["model_mae"] = round(mae, 2)
    return jsonify(result)

@app.route('/api/feedback', methods=['POST'])
def feedback():
    global current_target
    data = request.json
    city = data.get('city', 'Delhi')
    feedback_type = data.get('feedback')
    
    if feedback_type == 'H':
        current_target -= 1.5
    elif feedback_type == 'C':
        current_target += 1.5
        
    result = optimize_hvac(current_target, city, env_model)
    if not result:
        return jsonify({"error": "Failed to fetch weather data."}), 400
        
    result["model_mae"] = round(mae, 2)
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
