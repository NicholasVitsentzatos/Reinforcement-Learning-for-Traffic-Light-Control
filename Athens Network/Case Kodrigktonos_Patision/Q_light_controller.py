import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

# SUMO Path Setup
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

# Define SUMO Config
Sumo_config = [
    'sumo-gui',
    '-c', 'Athens Network\Case Kodrigktonos_Patision\Data\osm.sumocfg',
    '--step-length', '0.10',
    '--delay', '50',
    '--lateral-resolution', '0'
]

# Start SUMO
traci.start(Sumo_config)
traci.gui.setSchema("View #0", "real world")

# Define Variables
q_EB_0 = q_EB_1 = q_EB_2 = 0
q_SB_0 = q_SB_1 = q_SB_2 = 0
current_phase = 0

TOTAL_STEPS = 5000
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
ACTIONS = [0, 1]
Q_table = {}
MIN_GREEN_STEPS = 100
last_switch_step = -MIN_GREEN_STEPS
traffic_light_id = "Kodrigktonos_Patision"

# Metrics tracking
STOP_SPEED_THRESHOLD = 0.1
vehicle_stop_counts = {}
vehicle_prev_stopped = {}

# Define Functions
def get_queue_length(detector_id):
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)

def get_max_Q_value_of_state(s):
    if s not in Q_table:
        Q_table[s] = np.zeros(len(ACTIONS))
    return np.max(Q_table[s])

def get_reward(state):
    return -float(sum(state[:-1]))  # Encourage shorter queues

def get_state():
    global q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, current_phase
    dets = ["Traffic Panel Detector 1", "Traffic Panel Detector 2", "Traffic Panel Detector 3",
            "Traffic Panel Detector 4", "Traffic Panel Detector 5", "Traffic Panel Detector 6"]
    q_EB_0, q_EB_1, q_EB_2 = [get_queue_length(d) for d in dets[:3]]
    q_SB_0, q_SB_1, q_SB_2 = [get_queue_length(d) for d in dets[3:]]
    current_phase = get_current_phase(traffic_light_id)
    return (q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, current_phase)

def apply_action(action, tls_id="Kodrigktonos_Patision"):
    global last_switch_step
    if action == 1 and current_simulation_step - last_switch_step >= MIN_GREEN_STEPS:
        program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        next_phase = (get_current_phase(tls_id) + 1) % len(program.phases)
        traci.trafficlight.setPhase(tls_id, next_phase)
        last_switch_step = current_simulation_step

def update_Q_table(old_state, action, reward, new_state):
    if old_state not in Q_table:
        Q_table[old_state] = np.zeros(len(ACTIONS))
    old_q = Q_table[old_state][action]
    best_future_q = get_max_Q_value_of_state(new_state)
    Q_table[old_state][action] = old_q + ALPHA * (reward + GAMMA * best_future_q - old_q)

def get_action_from_policy(state):
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    else:
        if state not in Q_table:
            Q_table[state] = np.zeros(len(ACTIONS))
        return int(np.argmax(Q_table[state]))

# Fully Online Continuous Learning Loop

# Lists to record data for plotting
step_history = []
reward_history = []
queue_history = []

waiting_time_history = []
stops_history = []
co2_history = []
fuel_history = []

cumulative_reward = 0.0

print("\n=== Starting Fully Online Continuous Learning ===")
for step in range(TOTAL_STEPS):
    current_simulation_step = step
    
    state = get_state()
    action = get_action_from_policy(state)
    apply_action(action)
    
    traci.simulationStep()  # Advance simulation
    
    new_state = get_state()
    reward = get_reward(new_state)
    cumulative_reward += reward

    update_Q_table(state, action, reward, new_state)

    # Track vehicle stops (per vehicle)
    for v in traci.vehicle.getIDList():
        speed = traci.vehicle.getSpeed(v)
        stopped = speed < STOP_SPEED_THRESHOLD
        if v not in vehicle_stop_counts:
            vehicle_stop_counts[v] = 0
            vehicle_prev_stopped[v] = False
        if stopped and not vehicle_prev_stopped[v]:
            vehicle_stop_counts[v] += 1
        vehicle_prev_stopped[v] = stopped

    # Record data every 100 steps
    if step % 100 == 0:
        step_history.append(step)
        reward_history.append(cumulative_reward)
        queue_history.append(sum(new_state[:-1]))

        # Metric calculations
        vehicles = traci.vehicle.getIDList()
        avg_waiting = sum(traci.vehicle.getWaitingTime(v) for v in vehicles) / len(vehicles) if vehicles else 0.0
        total_stops = sum(vehicle_stop_counts.values())
        total_co2 = sum(traci.vehicle.getCO2Emission(v) for v in vehicles)
        total_fuel = sum(traci.vehicle.getFuelConsumption(v) for v in vehicles)

        waiting_time_history.append(avg_waiting)
        stops_history.append(total_stops)
        co2_history.append(total_co2)
        fuel_history.append(total_fuel)

        print(f"[Step {step}] Reward: {reward:.2f}, Cumulative: {cumulative_reward:.2f}")#, Q: {Q_table[state]}")
        print(f"Queue: {sum(new_state[:-1])}, Avg Wait: {avg_waiting:.2f}s, Stops: {total_stops}, CO₂: {total_co2:.2f}, Fuel: {total_fuel:.2f}")

# Close SUMO
traci.close()

print("\n========== Simulation Summary ==========")
print(f"Cumulative Reward: {reward_history[-1]:.2f}")
print(f"Average Queue Length: {sum(queue_history) / len(queue_history):.2f}")
print(f"Average Waiting Time: {sum(waiting_time_history) / len(waiting_time_history):.2f} s")
print(f"Average Number of Stops: {sum(stops_history) / len(stops_history):.2f}")
print(f"Average CO2 Emissions: {sum(co2_history) / len(co2_history):.2f} mg")
print(f"Average Fuel Consumption: {sum(fuel_history) / len(fuel_history):.2f} ml")
print("========================================\n")

print("\nTraining completed. Final Q-table size:", len(Q_table))
# for st, actions in Q_table.items():
#     print("State:", st, "-> Q-values:", actions)

# Plotting
metrics = [
    (reward_history, "Cumulative Reward", "Cumulative Reward", 'blue'),
    (queue_history, "Total Queue Length", "Vehicles", 'orange'),
    (waiting_time_history, "Average Waiting Time", "Seconds", 'green'),
    (stops_history, "Total Number of Stops", "Stops", 'red'),
    (co2_history, "Total CO₂ Emissions", "mg", 'purple'),
    (fuel_history, "Total Fuel Consumption", "ml", 'brown'),
]

plt.figure(figsize=(12, 8))

for i, (history, title, ylabel, color) in enumerate(metrics, 1):
    plt.subplot(3, 2, i)
    plt.plot(step_history, history, color=color)
    plt.xlabel("Simulation Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

plt.suptitle("Q Traffic Light Metrics Over Simulation Steps", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig(r"Athens Network\Case Kodrigktonos_Patision\Results\Q Traffic Light\Combined_Metrics.png", dpi=300, bbox_inches='tight')
plt.show()