import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

# Set SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Import TraCI
import traci

# Define SUMO config
Sumo_config = [
    'sumo-gui',
    '-c', 'Athens Network\Case Kodrigktonos_Patision\Data\osm.sumocfg',
    '--step-length', '0.10',
    '--delay', '0',
    '--lateral-resolution', '0'
]

# Start SUMO
traci.start(Sumo_config)
traci.gui.setSchema("View #0", "real world")

# Simulation parameters
TOTAL_STEPS = 5000
traffic_light_id = "Kodrigktonos_Patision"
STOP_SPEED_THRESHOLD = 0.1  # m/s — anything below this is considered a stop

# Data containers
step_history = []
reward_history = []
queue_history = []
waiting_time_history = []
stops_history = []
co2_history = []
fuel_history = []

# Vehicle tracking for stop counting
vehicle_stop_counts = {}
vehicle_prev_stopped = {}

# --------------------------
# Metric Functions
# --------------------------

def get_queue_length(detector_id):
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)

def get_total_queue():
    detectors = [
        "Traffic Panel Detector 1", "Traffic Panel Detector 2", "Traffic Panel Detector 3",
        "Traffic Panel Detector 4", "Traffic Panel Detector 5", "Traffic Panel Detector 6"
    ]
    return sum(get_queue_length(d) for d in detectors)

def get_avg_waiting_time():
    vehicles = traci.vehicle.getIDList()
    if not vehicles:
        return 0.0
    wait_times = [traci.vehicle.getWaitingTime(v) for v in vehicles]
    return sum(wait_times) / len(vehicles)

def get_total_co2():
    return sum(traci.vehicle.getCO2Emission(v) for v in traci.vehicle.getIDList())

def get_total_fuel():
    return sum(traci.vehicle.getFuelConsumption(v) for v in traci.vehicle.getIDList())

def get_total_stops():
    return sum(vehicle_stop_counts.values())

def get_reward():
    return -get_total_queue()  # Simple queue-based reward

# --------------------------
# Main Simulation Loop
# --------------------------

cumulative_reward = 0.0

print("\n=== Starting Simulation with Metrics ===")
for step in range(TOTAL_STEPS):
    current_simulation_step = step

    traci.simulationStep()  # Advance SUMO by one step

    # Stop tracking based on speed transitions
    vehicles = traci.vehicle.getIDList()
    for v in vehicles:
        speed = traci.vehicle.getSpeed(v)
        stopped = speed < STOP_SPEED_THRESHOLD

        if v not in vehicle_stop_counts:
            vehicle_stop_counts[v] = 0
            vehicle_prev_stopped[v] = False

        if stopped and not vehicle_prev_stopped[v]:
            vehicle_stop_counts[v] += 1

        vehicle_prev_stopped[v] = stopped

    # Get all metrics
    total_queue = get_total_queue()
    avg_wait = get_avg_waiting_time()
    total_stops = get_total_stops()
    co2 = get_total_co2()
    fuel = get_total_fuel()
    reward = get_reward()
    cumulative_reward += reward

    # Save every 100 steps
    if step % 100 == 0:
        step_history.append(step)
        reward_history.append(cumulative_reward)
        queue_history.append(total_queue)
        waiting_time_history.append(avg_wait)
        stops_history.append(total_stops)
        co2_history.append(co2)
        fuel_history.append(fuel)
        print(f"Step {step} | Cumulative: {cumulative_reward:.2f} | Queue: {total_queue} | Avg Wait: {avg_wait:.2f} s | Stops: {total_stops} | CO₂: {co2:.2f} mg | Fuel: {fuel:.2f} ml")

# --------------------------
# Close Simulation
# --------------------------
traci.close()
print("Simulation completed.")

print("\n========== Simulation Summary ==========")
print(f"Cumulative Reward: {reward_history[-1]:.2f}")
print(f"Average Queue Length: {sum(queue_history) / len(queue_history):.2f}")
print(f"Average Waiting Time: {sum(waiting_time_history) / len(waiting_time_history):.2f} s")
print(f"Average Number of Stops: {sum(stops_history) / len(stops_history):.2f}")
print(f"Average CO2 Emissions: {sum(co2_history) / len(co2_history):.2f} mg")
print(f"Average Fuel Consumption: {sum(fuel_history) / len(fuel_history):.2f} ml")
print("========================================\n")

# --------------------------
# Plot Metrics
# --------------------------

metrics = [
    (reward_history, "Cumulative Reward", "Cumulative Reward", 'blue'),
    (queue_history, "Total Queue Length", "Vehicles", 'orange'),
    (waiting_time_history, "Average Waiting Time", "Seconds", 'green'),
    (stops_history, "Number of Stops", "Stops", 'red'),
    (co2_history, "CO₂ Emissions", "mg", 'purple'),
    (fuel_history, "Fuel Consumption", "ml", 'brown'),
]

plt.figure(figsize=(12, 8))

for i, (history, title, ylabel, color) in enumerate(metrics, 1):
    plt.subplot(3, 2, i)
    plt.plot(step_history, history, color=color)
    plt.xlabel("Simulation Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

plt.suptitle("Typical Traffic Light Metrics Over Simulation Steps", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig(r"Athens Network\Case Kodrigktonos_Patision\Results\Typical Traffic Light\Combined_Metrics.png", dpi=300, bbox_inches='tight')
plt.show()