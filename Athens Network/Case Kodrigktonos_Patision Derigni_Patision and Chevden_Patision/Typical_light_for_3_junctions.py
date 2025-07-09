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
    '-c', 'Athens Network\Data\osm.sumocfg',
    '--step-length', '0.10',
    '--delay', '0',
    '--lateral-resolution', '0'
]

# Start SUMO
traci.start(Sumo_config)
traci.gui.setSchema("View #0", "real world")

# Simulation parameters
TOTAL_STEPS = 5000
# Traffic lights
traffic_light_id_kodrigktonos = "Kodrigktonos_Patision"
traffic_light_id_derigni = "Derigni_Patision"
traffic_light_id_cheven = "Cheven_Patision"
STOP_SPEED_THRESHOLD = 0.1  # m/s — anything below this is considered a stop

# Data containers
step_history = []
reward_history = []
queue_history_kodrigktonos = []
queue_history_derigni = []
queue_history_cheven = []
queue_history_total = []
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

def get_total_queue(detectors):
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

def get_reward(total_queue):
    return -total_queue  # Simple queue-based reward

# --------------------------
# Detector Lists for Junctions
# --------------------------

detectors_kodrigktonos = [
    "Traffic Panel Detector 1", "Traffic Panel Detector 2", "Traffic Panel Detector 3",
    "Traffic Panel Detector 4", "Traffic Panel Detector 5", "Traffic Panel Detector 6"
]

detectors_derigni = [
    "Traffic Panel Detector 7", "Traffic Panel Detector 8", "Traffic Panel Detector 9",
    "Traffic Panel Detector 10", "Traffic Panel Detector 11"
]

detectors_cheven = [
    "Traffic Panel Detector 12", "Traffic Panel Detector 13",
    "Traffic Panel Detector 14", "Traffic Panel Detector 15"
]

# --------------------------
# Main Simulation Loop
# --------------------------

cumulative_reward = 0.0

print("\n=== Starting Simulation with Metrics for Three Junctions ===")
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

    # Get metrics for all junctions
    total_queue_kodrigktonos = get_total_queue(detectors_kodrigktonos)
    total_queue_derigni = get_total_queue(detectors_derigni)
    total_queue_cheven = get_total_queue(detectors_cheven)
    total_queue = total_queue_kodrigktonos + total_queue_derigni + total_queue_cheven

    avg_wait = get_avg_waiting_time()
    total_stops = get_total_stops()
    co2 = get_total_co2()
    fuel = get_total_fuel()
    reward = get_reward(total_queue)
    cumulative_reward += reward

    # Save every 100 steps
    if step % 100 == 0:
        step_history.append(step)
        reward_history.append(cumulative_reward)
        queue_history_kodrigktonos.append(total_queue_kodrigktonos)
        queue_history_derigni.append(total_queue_derigni)
        queue_history_cheven.append(total_queue_cheven)
        queue_history_total.append(total_queue)
        waiting_time_history.append(avg_wait)
        stops_history.append(total_stops)
        co2_history.append(co2)
        fuel_history.append(fuel)
        
        print(
            f"\nStep {step} Metrics:"
            f"\n  Cumulative Reward: {cumulative_reward:.2f}  |  Total Queue: {total_queue}"
            f"\n  Queues -> Kodrigktonos: {total_queue_kodrigktonos}, Derigni: {total_queue_derigni}, Cheven: {total_queue_cheven}"
            f"\n  Total Queue: {total_queue}"
            f"\n  Avg. Waiting Time: {avg_wait:.2f}s  |  Total Stops: {total_stops}"
            f"\n  CO₂ Emissions: {co2:.2f}g  |  Fuel Consumption: {fuel:.2f}L"
        )

# --------------------------
# Close Simulation
# --------------------------
traci.close()
print("Simulation completed.")

print("\n===== Simulation Summary =====")
print(f"Cumulative Reward: {reward_history[-1]:.2f}")

print(f"Average Queue Length (Kodrigktonos): {sum(queue_history_kodrigktonos) / len(queue_history_kodrigktonos):.2f}")
print(f"Average Queue Length (Derigni): {sum(queue_history_derigni) / len(queue_history_derigni):.2f}")
print(f"Average Queue Length (Cheven): {sum(queue_history_cheven) / len(queue_history_cheven):.2f}")
print(f"Average Queue Length (Total): {sum(queue_history_total) / len(queue_history_total):.2f}")

print(f"Average Waiting Time: {sum(waiting_time_history) / len(waiting_time_history):.2f} s")
print(f"Average Stops: {sum(stops_history) / len(stops_history):.2f}")
print(f"Average CO2 Emissions: {sum(co2_history) / len(co2_history):.2f} mg")
print(f"Average Fuel Consumption: {sum(fuel_history) / len(fuel_history):.2f} ml")
print("====================================\n")


# --------------------------
# Plot Metrics
# --------------------------

plt.figure(figsize=(14, 12))

plt.subplot(3, 2, 1)
plt.plot(step_history, reward_history, label="Cumulative Reward", color='blue')
plt.xlabel("Simulation Step")
plt.ylabel("Reward")
plt.title("Cumulative Reward")
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(step_history, queue_history_kodrigktonos, label="Kodrigktonos")
plt.plot(step_history, queue_history_derigni, label="Derigni")
plt.plot(step_history, queue_history_cheven, label="Cheven")
plt.plot(step_history, queue_history_total, label="Total")
plt.xlabel("Simulation Step")
plt.ylabel("Queue Length")
plt.title("Queue Length per Junction")
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 3)
plt.plot(step_history, waiting_time_history, label="Avg Waiting Time", color='green')
plt.xlabel("Simulation Step")
plt.ylabel("Seconds")
plt.title("Average Waiting Time")
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(step_history, stops_history, label="Stops", color='red')
plt.xlabel("Simulation Step")
plt.ylabel("Stops")
plt.title("Number of Stops")
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 5)
plt.plot(step_history, co2_history, label="CO₂ Emissions", color='purple')
plt.xlabel("Simulation Step")
plt.ylabel("mg")
plt.title("CO₂ Emissions")
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 6)
plt.plot(step_history, fuel_history, label="Fuel Consumption", color='brown')
plt.xlabel("Simulation Step")
plt.ylabel("ml")
plt.title("Fuel Consumption")
plt.grid(True)
plt.legend()




plt.suptitle("Typical Traffic Lights Metrics Over Simulation Steps", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig(r"Athens Network\Case Kodrigktonos_Patision Derigni_Patision and Chevden_Patision\Results\Typical Traffic Lights\Combined_Metrics.png", dpi=300, bbox_inches='tight')
plt.show()