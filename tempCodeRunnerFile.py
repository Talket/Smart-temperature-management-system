import requests

def get_live_weather(city_name):
    # Step 1: Use Geocoding API to find the Latitude and Longitude of the city
    # We need coordinates because the weather API relies on exact GPS locations
    geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
    
    try:
        geo_response = requests.get(geocode_url)
        geo_data = geo_response.json()
        
        # Check if the city was actually found
        if "results" not in geo_data or len(geo_data["results"]) == 0:
            print(f"Error: Could not find coordinates for '{city_name}'.")
            return None
            
        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]
        country = geo_data["results"][0].get("country", "Unknown")
        
        # Step 2: Use the coordinates to fetch the current temperature and humidity
        # We request 'temperature_2m' and 'relative_humidity_2m' as they are required for our ML model
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m"
        
        weather_response = requests.get(weather_url)
        weather_data = weather_response.json()
        
        current_weather = weather_data["current"]
        
        # Return a clean dictionary with the data we need for Step 2
        return {
            "city": city_name.capitalize(),
            "country": country,
            "temperature_c": current_weather["temperature_2m"],
            "humidity_percent": current_weather["relative_humidity_2m"]
        }
        
    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")
        return None

if __name__ == "__main__":
    # Test with a few different Indian cities
    test_cities = ["Delhi", "Chennai", "Ludhiana"]
    
    for city in test_cities:
        weather = get_live_weather(city)
        if weather:
            print(f"--- {weather['city']}, {weather['country']} ---")
            print(f"Temperature: {weather['temperature_c']}°C")
            print(f"Humidity: {weather['humidity_percent']}%\n")