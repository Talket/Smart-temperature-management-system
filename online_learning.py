import torch
import torch.optim as optim

# Directly import the required modules
from weather_api import get_live_weather
from environment_predictor import RoomTemperaturePredictor, generate_room_data, train_and_evaluate

def interactive_feedback_loop(city_name, model):
    """
    Simulates Online Learning by allowing the user to provide real-time feedback.
    The optimizer recalculates the global minima based on dynamic user comfort shifts.
    """
    print(f"\n--- Initializing Interactive Optimization for {city_name} ---")
    
    # Fetch live environmental data once for the session
    weather = get_live_weather(city_name)
    if not weather:
        return
    t_out = weather['temperature_c']
    humidity = weather['humidity_percent']
    
    # Freeze model weights for inference
    for param in model.parameters():
        param.requires_grad = False
        
    # Standard starting comfort target
    current_target = 24.0 
    
    # Interactive Loop
    while True:
        print(f"\n[ SYSTEM ] Current Comfort Target: {current_target}°C")
        
        # Reset settings tensor for fresh optimization cycle
        settings = torch.tensor([30.0, 10.0, 0.0], requires_grad=True)
        optimizer = optim.Adam([settings], lr=0.3)
        
        # Run Gradient Descent
        for epoch in range(150):
            optimizer.zero_grad()
            
            env_tensor = torch.tensor([t_out, humidity], dtype=torch.float32)
            combined_inputs = torch.cat((env_tensor, settings)).unsqueeze(0)
            
            predicted_temp = model(combined_inputs)[0][0]
            
            # Loss Function using the DYNAMIC target
            comfort_loss = (predicted_temp - current_target) ** 2
            energy_loss = (torch.relu(28.0 - settings[0]) * 0.5) + \
                          (torch.relu(settings[1] - 15.0) * 0.6) + \
                          (settings[2] * 0.1)
                          
            loss = comfort_loss + (0.3 * energy_loss)
            
            loss.backward()
            optimizer.step()
            
            # Constraints
            with torch.no_grad():
                settings[0].clamp_(16.0, 30.0)
                settings[1].clamp_(10.0, 30.0)
                settings[2].clamp_(0.0, 5.0)

        # Extract optimal values
        opt_ac, opt_heater, opt_fan = settings.detach().numpy()
        pred_temp = model(torch.cat((torch.tensor([t_out, humidity], dtype=torch.float32), settings)).unsqueeze(0))[0][0].item()
        
        # Format display (hide if inactive)
        ac_disp = f"{opt_ac:.1f}°C" if opt_ac < t_out else "OFF"
        heat_disp = f"{opt_heater:.1f}°C" if opt_heater > t_out else "OFF"
        fan_disp = f"Level {opt_fan:.1f}" if opt_fan > 0.1 else "OFF"
        
        print(f"Optimal Settings -> AC: {ac_disp} | Heater: {heat_disp} | Fan: {fan_disp}")
        print(f"Predicted Room Temp: {pred_temp:.1f}°C")
        
        # --- THE FEEDBACK LOOP ---
        feedback = input("\nProvide Feedback - [H] Too Hot, [C] Too Cold, [P] Perfect: ").strip().upper()
        
        if feedback == 'P':
            print("System optimized to your personal profile. Exiting loop.")
            break
        elif feedback == 'H':
            print(">> Adjusting system weights for cooler preference...")
            current_target -= 1.5 # Lower the target target
        elif feedback == 'C':
            print(">> Adjusting system weights for warmer preference...")
            current_target += 1.5 # Raise the target temp
        else:
            print("Invalid input. Please enter H, C, or P.")

if __name__ == "__main__":
    # Quickly prepare the environment model
    X_data, y_data = generate_room_data(2000)
    env_model = RoomTemperaturePredictor()
    env_model = train_and_evaluate(env_model, X_data, y_data, epochs=150)
    
    # Start the interactive feedback loop
    interactive_feedback_loop("Delhi", env_model)