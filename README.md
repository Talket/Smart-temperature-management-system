# Adaptive Room Climate Optimizer

An experimental machine-learning project that explores how a room's temperature can be predicted and how HVAC settings can be adjusted toward a user-defined comfort target.

> **Project status:** Educational prototype / experiment. The model is trained on generated synthetic room data; it has not been validated as a real building-control system and does not directly actuate physical HVAC equipment.

## What it explores

- A small PyTorch feed-forward network that predicts room temperature from outdoor temperature, humidity, AC setpoint, heater setpoint, and fan speed.
- Live outdoor temperature and humidity lookup using the free Open-Meteo geocoding and forecast APIs.
- Input optimization with PyTorch's Adam optimizer to search for HVAC settings that move the model's predicted room temperature toward a target.
- A Flask API and a browser-based interface for interacting with the experiment.

## How the pieces fit

1. **Weather lookup** — resolve a city with Open-Meteo and retrieve current weather conditions.
2. **Synthetic data generation** — generate simulated room conditions and corresponding temperature targets.
3. **Model training** — train a compact neural network to approximate the generated room-temperature relationship.
4. **Prediction and optimization** — freeze model weights and optimize HVAC input settings against the requested comfort target.
5. **Interface** — send requests to the Flask service and display the experiment results.

## Tech stack

- Python
- PyTorch
- NumPy
- Flask and Flask-CORS
- Requests
- Open-Meteo API
- HTML, CSS and JavaScript

## Run locally

Use Python 3.10 or newer.

```bash
git clone https://github.com/Talket/Smart-temperature-management-system.git
cd Smart-temperature-management-system

python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

pip install numpy requests flask flask-cors
```

Install PyTorch using the instructions for your operating system and hardware at https://pytorch.org/get-started/locally/.

Start the API:

```bash
python server.py
```

Then open the project's `index.html` in a browser. The interface expects the local API configured by the application.

## Repository map

| File | Responsibility |
|---|---|
| `server.py` | Flask API and model initialization |
| `environment_predictor.py` | Synthetic data generation, model definition and training |
| `weather_api.py` | Open-Meteo geocoding and current-weather lookup |
| `optimization.py` | Gradient-based search for HVAC input settings |
| `online_learning.py` | Interactive feedback-loop experiment |
| `main_app.py` | Combined command-line experiment |
| `index.html` | Browser interface |

## Limitations and next steps

- Synthetic training data is a simplified simulation, not measured room telemetry.
- Prediction quality must be tested on real, held-out sensor data before drawing practical conclusions.
- Network requests need explicit timeouts and stronger error handling for production use.
- The current optimizer proposes values for the model; it does not control equipment or enforce hardware safety constraints.
- A useful next step is a reproducible evaluation notebook with baseline comparisons, error metrics, and real sensor data collected with consent.

## Disclaimer

This repository is for learning and experimentation. Do not use its predictions to control real heating, ventilation, or air-conditioning equipment.
