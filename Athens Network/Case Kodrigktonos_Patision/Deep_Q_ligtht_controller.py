import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Set SUMO Environment
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

# Define SUMO configuration
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
MIN_GREEN_STEPS = 100
last_switch_step = -MIN_GREEN_STEPS

# Metrics-related variables
STOP_SPEED_THRESHOLD = 0.1
vehicle_stop_counts = {}
vehicle_prev_stopped = {}

# Build DQN model
def build_model(state_size, action_size):
    model = keras.Sequential([
        keras.layers.Input(shape=(state_size,)),
        keras.layers.Dense(24, activation='relu'),
        keras.layers.Dense(24, activation='relu'),
        keras.layers.Dense(action_size, activation='linear')
    ])
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001))
    return model

def to_array(state):
    return np.array(state, dtype=np.float32).reshape((1, -1))

state_size = 7
action_size = len(ACTIONS)
dqn_model = build_model(state_size, action_size)

# RL-related functions
def get_state():
    global q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, current_phase
    dets = ["Traffic Panel Detector 1", "Traffic Panel Detector 2", "Traffic Panel Detector 3",
            "Traffic Panel Detector 4", "Traffic Panel Detector 5", "Traffic Panel Detector 6"]
    q_EB_0, q_EB_1, q_EB_2 = [traci.lanearea.getLastStepVehicleNumber(d) for d in dets[:3]]
    q_SB_0, q_SB_1, q_SB_2 = [traci.lanearea.getLastStepVehicleNumber(d) for d in dets[3:]]
    current_phase = traci.trafficlight.getPhase("Kodrigktonos_Patision")
    return (q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, current_phase)

def get_reward(state):
    return -float(sum(state[:-1]))

def apply_action(action):
    global last_switch_step
    if action == 1 and current_simulation_step - last_switch_step >= MIN_GREEN_STEPS:
        prog = traci.trafficlight.getAllProgramLogics("Kodrigktonos_Patision")[0]
        next_phase = (traci.trafficlight.getPhase("Kodrigktonos_Patision") + 1) % len(prog.phases)
        traci.trafficlight.setPhase("Kodrigktonos_Patision", next_phase)
        last_switch_step = current_simulation_step

def update_Q_table(old_state, action, reward, new_state):
    old_arr = to_array(old_state)
    Q_old = dqn_model.predict(old_arr, verbose=0)[0]
    new_arr = to_array(new_state)
    Q_new = dqn_model.predict(new_arr, verbose=0)[0]
    Q_old[action] = Q_old[action] + ALPHA * (reward + GAMMA * np.max(Q_new) - Q_old[action])
    dqn_model.fit(old_arr, np.array([Q_old]), verbose=0)

def get_action_from_policy(state):
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    q_vals = dqn_model.predict(to_array(state), verbose=0)[0]
    return int(np.argmax(q_vals))

# Metric helper functions
def get_avg_waiting_time():
    vehicles = traci.vehicle.getIDList()
    return sum(traci.vehicle.getWaitingTime(v) for v in vehicles) / len(vehicles) if vehicles else 0.0

def get_total_co2():
    return sum(traci.vehicle.getCO2Emission(v) for v in traci.vehicle.getIDList())

def get_total_fuel():
    return sum(traci.vehicle.getFuelConsumption(v) for v in traci.vehicle.getIDList())

def update_vehicle_stops():
    """
    Update stop counts per vehicle based on speed threshold.
    If a vehicle's speed falls below STOP_SPEED_THRESHOLD from above, increment its stop count.
    """
    global vehicle_stop_counts, vehicle_prev_stopped
    current_vehicles = traci.vehicle.getIDList()

    # Remove vehicles no longer in the simulation
    for v in list(vehicle_stop_counts.keys()):
        if v not in current_vehicles:
            vehicle_stop_counts.pop(v)
            vehicle_prev_stopped.pop(v, None)

    for v in current_vehicles:
        speed = traci.vehicle.getSpeed(v)
        prev_stopped = vehicle_prev_stopped.get(v, False)

        if speed < STOP_SPEED_THRESHOLD:
            # Vehicle is currently stopped
            if not prev_stopped:
                # Vehicle just stopped now
                vehicle_stop_counts[v] = vehicle_stop_counts.get(v, 0) + 1
            vehicle_prev_stopped[v] = True
        else:
            vehicle_prev_stopped[v] = False

def get_total_stops():
    return sum(vehicle_stop_counts.values())

# Online Training Loop with Metrics
step_history = []
reward_history = []
queue_history = []
avg_wait_time_history = []
total_stops_history = []
co2_history = []
fuel_history = []

cumulative_reward = 0.0

print("\n=== Starting Fully Online Continuous Learning (DQN) with Metrics ===")

for step in range(TOTAL_STEPS):
    current_simulation_step = step

    state = get_state()
    action = get_action_from_policy(state)
    apply_action(action)

    traci.simulationStep()  # Advance one step

    new_state = get_state()
    reward = get_reward(new_state)
    cumulative_reward += reward

    update_Q_table(state, action, reward, new_state)
    update_vehicle_stops()

    if step % 100 == 0 or step == TOTAL_STEPS - 1:

        avg_wait = get_avg_waiting_time()
        total_stops = get_total_stops()
        total_co2 = get_total_co2()
        total_fuel = get_total_fuel()

        step_history.append(step)
        reward_history.append(cumulative_reward)
        queue_history.append(sum(new_state[:-1]))
        avg_wait_time_history.append(avg_wait)
        total_stops_history.append(total_stops)
        co2_history.append(total_co2)
        fuel_history.append(total_fuel)

        q_vals = dqn_model.predict(to_array(state), verbose=0)[0]
        print(f"Step {step}, Action: {action}, Reward: {reward:.2f}, Cumulative Reward: {cumulative_reward:.2f}")
        print(f"Queues: {sum(new_state[:-1])}, Avg Wait: {avg_wait:.2f}, Stops: {total_stops}, CO2: {total_co2:.2f}, Fuel: {total_fuel:.2f}")
        # print(f"Q-values: {q_vals}")

# Close SUMO 
traci.close()

print("\nOnline Training completed.")
print("\n========== Simulation Summary ==========")
print(f"Cumulative Reward: {reward_history[-1]:.2f}")
print(f"Average Queue Length: {sum(queue_history) / len(queue_history):.2f}")
print(f"Average Waiting Time: {sum(avg_wait_time_history) / len(avg_wait_time_history):.2f} s")
print(f"Average Number of Stops: {sum(total_stops_history) / len(total_stops_history):.2f}")
print(f"Average CO2 Emissions: {sum(co2_history) / len(co2_history):.2f} mg")
print(f"Average Fuel Consumption: {sum(fuel_history) / len(fuel_history):.2f} ml")
print("========================================\n")

print("DQN Model Summary:")
dqn_model.summary()

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(3, 2, 1)
plt.plot(step_history, reward_history)
plt.xlabel("Step")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward Over Time")
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(step_history, queue_history, color='orange')
plt.xlabel("Step")
plt.ylabel("Queue Length")
plt.title("Total Queue Length")
plt.grid(True)

plt.subplot(3, 2, 3)
plt.plot(step_history, avg_wait_time_history, color='green')
plt.xlabel("Step")
plt.ylabel("Avg Waiting Time (s)")
plt.title("Average Waiting Time")
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(step_history, total_stops_history, color='red')
plt.xlabel("Step")
plt.ylabel("Number of Stops")
plt.title("Total Stops")
plt.grid(True)

plt.subplot(3, 2, 5)
plt.plot(step_history, co2_history, color='purple')
plt.xlabel("Step")
plt.ylabel("CO2 Emissions")
plt.title("Total CO2 Emissions")
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(step_history, fuel_history, color='brown')
plt.xlabel("Step")
plt.ylabel("Fuel Consumption")
plt.title("Fuel Consumption")
plt.grid(True)

plt.suptitle("DQN Traffic Light Metrics Over Simulation Steps")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig(r"Athens Network\Case Kodrigktonos_Patision\Results\Deep Q Traffic Light\Combined_Metrics.png", dpi=300, bbox_inches='tight')
plt.show()
