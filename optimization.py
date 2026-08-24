import torch
import torch.optim as optim

# Direct imports from your specific files
from weather_api import get_live_weather
from environment_predictor import RoomTemperaturePredictor, generate_room_data, train_and_evaluate

def optimize_hvac(target_temp, city_name, model):
    print(f"\n--- Running Optimization Engine for {city_name} ---")
    print(f"Goal: Achieve {target_temp}°C with minimum energy consumption.")
    
    # 1. Fetch live weather
    weather = get_live_weather(city_name)
    if not weather:
        return
    t_out = weather['temperature_c']
    humidity = weather['humidity_percent']
    print(f"Current Outdoor Weather: {t_out}°C, {humidity}% Humidity")
    
    # 2. Freeze the model's weights. We are optimizing the INPUTS, not the model.
    for param in model.parameters():
        param.requires_grad = False
        
    # 3. Initial Guess: AC Off (30°C), Heater Off (10°C), Fan Off (0)
    # We set requires_grad=True because these are the variables PyTorch will optimize
    settings = torch.tensor([30.0, 10.0, 0.0], requires_grad=True)
    
    # Use the Adam Optimizer with a learning rate of 0.2
    optimizer = optim.Adam([settings], lr=0.2)
    
    print("\nStarting Gradient Descent...")
    epochs = 200
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Combine fixed weather with the adjustable settings
        # [OutdoorTemp, Humidity, ACTemp, HeaterTemp, FanSpeed]
        env_tensor = torch.tensor([t_out, humidity], dtype=torch.float32)
        combined_inputs = torch.cat((env_tensor, settings)).unsqueeze(0)
        
        # Predict the resulting room temperature
        predicted_temp = model(combined_inputs)[0][0]
        
        # --- THE OBJECTIVE / LOSS FUNCTION ---
        
        # A. Comfort Loss (Mean Squared Error between predicted temp and target temp)
        comfort_loss = (predicted_temp - target_temp) ** 2
        
        # B. Energy Loss (A differentiable proxy function for electricity cost)
        # AC uses energy if set below 28, Heater if set above 15, Fan uses linear energy
        energy_loss = (torch.relu(28.0 - settings[0]) * 0.5) + \
                      (torch.relu(settings[1] - 15.0) * 0.6) + \
                      (settings[2] * 0.1)
                      
        # Total Loss (Balancing Comfort vs Energy. 0.3 is the energy penalty weight)
        loss = comfort_loss + (0.3 * energy_loss)
        
        # Calculate gradients
        loss.backward()
        
        # Update the settings based on the gradients
        optimizer.step()
        
        # Apply strict physical constraints (can't set AC to -500 degrees)
        with torch.no_grad():
            settings[0].clamp_(16.0, 30.0) # AC limits
            settings[1].clamp_(10.0, 30.0) # Heater limits
            settings[2].clamp_(0.0, 5.0)   # Fan limits
            
        if (epoch + 1) % 50 == 0:
            print(f"Iteration {epoch+1:03d} | Loss: {loss.item():.2f} | "
                  f"Predicted Temp: {predicted_temp.item():.2f}°C")

    # Final Output
    final_ac, final_heater, final_fan = settings.detach().numpy()
    final_pred = model(torch.cat((torch.tensor([t_out, humidity], dtype=torch.float32), settings)).unsqueeze(0))[0][0].item()
    
    print("\n--- GLOBAL MINIMA FOUND (Optimal Settings) ---")
    print(f"AC Temperature: {final_ac:.1f}°C")
    print(f"Heater Temperature: {final_heater:.1f}°C")
    print(f"Fan Speed Level: {final_fan:.1f}")
    print(f"Resulting Room Temp: {final_pred:.2f}°C (Target was {target_temp}°C)")
    
    return {
        "outdoor_temp": round(t_out, 1),
        "humidity": round(humidity, 1),
        "ac_temp": f"{final_ac:.1f}°C" if final_ac < t_out else "OFF",
        "heater_temp": f"{final_heater:.1f}°C" if final_heater > t_out else "OFF",
        "fan_speed": f"Level {final_fan:.1f}" if final_fan > 0.1 else "OFF",
        "predicted_temp": round(final_pred, 1),
        "target_temp": target_temp,
        "city": city_name,
        "country": "India"
    }

if __name__ == "__main__":
    # We must quickly train the environment model in memory first so the optimizer has a brain to use
    print("Initializing System...")
    X_data, y_data = generate_room_data(2000)
    env_model = RoomTemperaturePredictor()
    env_model = train_and_evaluate(env_model, X_data, y_data, epochs=150)
    
    # Get city input from the user
    user_city = input("\nEnter your city name (e.g., Delhi, Mumbai, Ludhiana): ").strip()
    if not user_city:
        user_city = "Delhi" # Fallback if user just hits Enter
        print("No city entered. Defaulting to Delhi.")
        
    # Run the Optimizer for the user's city and a cold target
    optimize_hvac(target_temp=24.0, city_name=user_city, model=env_model)