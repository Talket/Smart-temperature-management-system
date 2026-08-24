import requests
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ==========================================
# 1. WEATHER API MODULE
# ==========================================
def get_live_weather(city_name):
    """Fetches real-time temperature and humidity using Open-Meteo API."""
    print(f"\n[ API ] Fetching live weather data for {city_name}...")
    geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
    
    try:
        geo_response = requests.get(geocode_url)
        geo_data = geo_response.json()
        
        if "results" not in geo_data or len(geo_data["results"]) == 0:
            print(f"[ API ERROR ] Could not find '{city_name}'. Using default weather.")
            return {"temperature_c": 35.0, "humidity_percent": 50.0}
            
        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]
        
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m"
        weather_data = requests.get(weather_url).json()
        
        return {
            "temperature_c": weather_data["current"]["temperature_2m"],
            "humidity_percent": weather_data["current"]["relative_humidity_2m"]
        }
    except Exception as e:
        print(f"[ API ERROR ] Network issue ({e}). Using default weather.")
        return {"temperature_c": 35.0, "humidity_percent": 50.0}

# ==========================================
# 2. ENVIRONMENT PREDICTOR MODULE
# ==========================================
def generate_room_data(num_samples=2000):
    """Generates synthetic physics data to train the Neural Network."""
    X, y = [], []
    for _ in range(num_samples):
        t_out = np.random.uniform(10.0, 45.0)
        humidity = np.random.uniform(20.0, 90.0)
        ac_set = np.random.uniform(16.0, 30.0) if np.random.rand() > 0.3 else 0.0
        heat_set = np.random.uniform(16.0, 30.0) if np.random.rand() > 0.7 else 0.0
        fan = np.random.uniform(0.0, 5.0)
        
        t_room = t_out
        if ac_set > 0 and ac_set < t_out: t_room -= (t_out - ac_set) * 0.75 
        elif heat_set > 0 and heat_set > t_out: t_room += (heat_set - t_out) * 0.75
        t_room -= (fan * 0.4)
        t_room += np.random.normal(0, 0.5)
        
        X.append([t_out, humidity, ac_set, heat_set, fan])
        y.append([t_room])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class RoomTemperaturePredictor(nn.Module):
    """Deep Learning Model simulating room thermodynamics."""
    def __init__(self):
        super(RoomTemperaturePredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x): return self.net(x)

def train_model(model, X, y, epochs=150):
    """Trains the predictive brain."""
    print("[ SYSTEM ] Training the Environment Prediction Model...")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    print(f"[ SYSTEM ] Training Complete. Final MSE Loss: {loss.item():.2f}")
    return model

# ==========================================
# 3. UNIFIED OPTIMIZATION & FEEDBACK LOOP
# ==========================================
def run_smart_hvac_system():
    print("="*50)
    print("   AI-POWERED HVAC OPTIMIZATION SYSTEM   ")
    print("="*50)
    
    # Step A: Initialize and Train Brain
    X_data, y_data = generate_room_data()
    model = train_model(RoomTemperaturePredictor(), X_data, y_data)
    for param in model.parameters(): param.requires_grad = False # Freeze weights
    
    # Step B: Get User Location
    city_name = input("\nEnter your city name (e.g., Delhi, Mumbai): ").strip() or "Delhi"
    
    # Step C: Fetch Real-Time Environment Constraints
    weather = get_live_weather(city_name)
    t_out = weather['temperature_c']
    humidity = weather['humidity_percent']
    
    current_target = 24.0 # Default comfort baseline
    
    # Step D: The Interactive Optimization Loop
    while True:
        print("\n" + "-"*40)
        print(f"[ DASHBOARD ] Outdoor Weather : {t_out}°C | Humidity: {humidity}%")
        print(f"[ DASHBOARD ] Target Comfort  : {current_target}°C")
        print("-" * 40)
        
        # Reset adjustable inputs
        settings = torch.tensor([30.0, 10.0, 0.0], requires_grad=True)
        optimizer = optim.Adam([settings], lr=0.3)
        
        print(">> Running Gradient Descent Optimizer...")
        for epoch in range(150):
            optimizer.zero_grad()
            
            # Combine fixed weather + variable settings
            env_tensor = torch.tensor([t_out, humidity], dtype=torch.float32)
            combined_inputs = torch.cat((env_tensor, settings)).unsqueeze(0)
            
            # Predict
            predicted_temp = model(combined_inputs)[0][0]
            
            # Calculate Objectives (Minimize Discomfort + Minimize Energy)
            comfort_loss = (predicted_temp - current_target) ** 2
            energy_loss = (torch.relu(28.0 - settings[0]) * 0.5) + \
                          (torch.relu(settings[1] - 15.0) * 0.6) + \
                          (settings[2] * 0.1)
            loss = comfort_loss + (0.3 * energy_loss)
            
            # Backpropagate to inputs
            loss.backward()
            optimizer.step()
            
            # Physical Constraints
            with torch.no_grad():
                settings[0].clamp_(16.0, 30.0)
                settings[1].clamp_(10.0, 30.0)
                settings[2].clamp_(0.0, 5.0)

        # Extract Optimal States
        opt_ac, opt_heat, opt_fan = settings.detach().numpy()
        pred_temp = model(torch.cat((torch.tensor([t_out, humidity], dtype=torch.float32), settings)).unsqueeze(0))[0][0].item()
        
        # Display Formatting
        ac_disp = f"{opt_ac:.1f}°C" if opt_ac < t_out else "OFF"
        heat_disp = f"{opt_heat:.1f}°C" if opt_heat > t_out else "OFF"
        fan_disp = f"Level {opt_fan:.1f}" if opt_fan > 0.1 else "OFF"
        
        print(f"\n[ RESULTS ] Optimal Settings Found!")
        print(f" -> AC Temp      : {ac_disp}")
        print(f" -> Heater Temp  : {heat_disp}")
        print(f" -> Fan Speed    : {fan_disp}")
        print(f" -> Predicted Room Temperature: {pred_temp:.2f}°C")
        
        # Step E: Online Learning Feedback Mechanism
        feedback = input("\nHow do you feel? [H] Too Hot | [C] Too Cold | [P] Perfect (Exit): ").strip().upper()
        
        if feedback == 'P':
            print("\nSystem successfully optimized to your personal profile. Goodbye!")
            break
        elif feedback == 'H':
            print(">> Updating AI preference... Shifting target cooler.")
            current_target -= 1.5 
        elif feedback == 'C':
            print(">> Updating AI preference... Shifting target warmer.")
            current_target += 1.5 
        else:
            print("Invalid input. Please enter H, C, or P.")

if __name__ == "__main__":
    run_smart_hvac_system()