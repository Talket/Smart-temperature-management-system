import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Directly import the live weather function
from weather_api import get_live_weather

# 1. Generate Synthetic Data for Room Thermodynamics
def generate_room_data(num_samples=5000):
    X = [] # Inputs: [OutdoorTemp, Humidity, ACTemp, HeaterTemp, FanSpeed]
    y = [] # Target: [ResultingRoomTemp]
    
    for _ in range(num_samples):
        # Random weather
        t_out = np.random.uniform(10.0, 45.0)
        humidity = np.random.uniform(20.0, 90.0)
        
        # Random settings (0 means OFF)
        ac_set = np.random.uniform(16.0, 30.0) if np.random.rand() > 0.3 else 0.0
        heat_set = np.random.uniform(16.0, 30.0) if np.random.rand() > 0.7 else 0.0
        fan = np.random.uniform(0.0, 5.0)
        
        # Physics simulation to determine actual room temperature
        t_room = t_out
        if ac_set > 0 and ac_set < t_out:
            # AC cools the room 75% of the way to the target temp
            t_room -= (t_out - ac_set) * 0.75 
        elif heat_set > 0 and heat_set > t_out:
            # Heater warms the room 75% of the way to target
            t_room += (heat_set - t_out) * 0.75
            
        t_room -= (fan * 0.4) # Fan reduces perceived temp
        
        # Add a tiny bit of random noise (0.5 degrees) for realism
        t_room += np.random.normal(0, 0.5)
        
        X.append([t_out, humidity, ac_set, heat_set, fan])
        y.append([t_room])
        
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# 2. Define the PyTorch Model
class RoomTemperaturePredictor(nn.Module):
    def __init__(self):
        super(RoomTemperaturePredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1) # Predicts a single value: Room Temp
        )

    def forward(self, x):
        return self.net(x)

# 3. Train the Model and Calculate Accuracy
def train_and_evaluate(model, X, y, epochs=300):
    print("--- Training Environment Predictor ---")
    
    # Split data: 80% for training, 20% for testing accuracy
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training Loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
        
    # Accuracy Check on unseen Test Data
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test)
        
        # Calculate Mean Absolute Error (MAE)
        mae = torch.mean(torch.abs(test_preds - y_test)).item()
        
        print("--- Training Complete ---")
        print(f"Model Accuracy (Mean Absolute Error): ±{mae:.2f}°C")
        print("This means predictions are generally within this many degrees of reality.\n")
        
        # Show a few test samples
        print("Sample Predictions on Test Data:")
        for i in range(3):
            actual = y_test[i].item()
            pred = test_preds[i].item()
            print(f"  Actual Room Temp: {actual:.1f}°C | Model Predicted: {pred:.1f}°C")
            
    return model

# 4. Integrate with Live Weather API
def run_live_prediction(model, city_name, ac_temp, heater_temp, fan_speed):
    print(f"\n--- Fetching Live Data for {city_name} ---")
    live_weather = get_live_weather(city_name)
    
    if not live_weather:
        return
        
    t_out = live_weather['temperature_c']
    humidity = live_weather['humidity_percent']
    
    # Create the input tensor for the model
    # [OutdoorTemp, Humidity, ACTemp, HeaterTemp, FanSpeed]
    current_state = torch.tensor([[t_out, humidity, ac_temp, heater_temp, fan_speed]], dtype=torch.float32)
    
    # Predict
    model.eval()
    with torch.no_grad():
        predicted_room_temp = model(current_state).item()
        
    print(f"Live Weather: {t_out}°C, {humidity}% Humidity")
    print(f"Appliance Settings -> AC: {ac_temp}°C, Heater: {heater_temp}°C, Fan: Level {fan_speed}")
    print(f">> PREDICTED ROOM TEMPERATURE: {predicted_room_temp:.2f}°C <<")

if __name__ == "__main__":
    # 1. Prepare data and model
    X_data, y_data = generate_room_data()
    predictor_model = RoomTemperaturePredictor()
    
    # 2. Train model and show accuracy
    predictor_model = train_and_evaluate(predictor_model, X_data, y_data)
    
    # 3. Test with a live city API request (Example: AC set to 22, Fan set to 3)
    run_live_prediction(predictor_model, "Ludhiana", ac_temp=22.0, heater_temp=0.0, fan_speed=3.0)
    run_live_prediction(predictor_model, "Delhi", ac_temp=24.0, heater_temp=0.0, fan_speed=5.0)