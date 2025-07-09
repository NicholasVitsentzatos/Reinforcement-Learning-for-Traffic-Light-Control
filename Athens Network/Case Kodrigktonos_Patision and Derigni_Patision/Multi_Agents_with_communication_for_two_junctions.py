import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

# Set SUMO Environment
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

# Define SUMO Configuration
sumo_config = [
    'sumo-gui',
    '-c', 'Athens Network\Data\osm.sumocfg',
    '--step-length', '0.10',
    '--delay', '50',
    '--lateral-resolution', '0'
]

# Start SUMO
traci.start(sumo_config)
traci.gui.setSchema("View #0", "real world")

# Hyperparameters
TOTAL_STEPS = 5000
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
ACTIONS = [0, 1]
MIN_GREEN_STEPS = 100

# Q-Tables
Q_table_KP = {}
Q_table_DP = {}
last_switch_step = {
    "Kodrigktonos_Patision": -MIN_GREEN_STEPS,
    "Derigni_Patision": -MIN_GREEN_STEPS
}

# Detector ID mappings
detectors = {
    "Kodrigktonos_Patision": ["Traffic Panel Detector 1", "Traffic Panel Detector 2", "Traffic Panel Detector 3",
                              "Traffic Panel Detector 4", "Traffic Panel Detector 5", "Traffic Panel Detector 6"],
    "Derigni_Patision": ["Traffic Panel Detector 7", "Traffic Panel Detector 8",
                         "Traffic Panel Detector 9", "Traffic Panel Detector 10", "Traffic Panel Detector 11"]
}

# RL-related functions 
def get_queue_length(detector_id):
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)

def get_joint_state():
    state_kp = [get_queue_length(det) for det in detectors["Kodrigktonos_Patision"]]
    state_dp = [get_queue_length(det) for det in detectors["Derigni_Patision"]]
    phase_kp = get_current_phase("Kodrigktonos_Patision")
    phase_dp = get_current_phase("Derigni_Patision")
    return tuple(state_kp + [phase_kp] + state_dp + [phase_dp])

def get_reward(state):
    return -float(sum(state[:-1]))

def get_action(agent_id, state, q_table):
    if state not in q_table:
        q_table[state] = np.zeros(len(ACTIONS))
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    return int(np.argmax(q_table[state]))

def update_q_table(state, action, reward, new_state, q_table):
    if state not in q_table:
        q_table[state] = np.zeros(len(ACTIONS))
    if new_state not in q_table:
        q_table[new_state] = np.zeros(len(ACTIONS))
    old_q = q_table[state][action]
    best_future_q = np.max(q_table[new_state])
    q_table[state][action] = old_q + ALPHA * (reward + GAMMA * best_future_q - old_q)

def apply_action(tls_id, action, current_step):
    if action == 1 and current_step - last_switch_step[tls_id] >= MIN_GREEN_STEPS:
        logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        num_phases = len(logic.phases)
        next_phase = (get_current_phase(tls_id) + 1) % num_phases
        traci.trafficlight.setPhase(tls_id, next_phase)
        last_switch_step[tls_id] = current_step

def get_vehicle_metrics(junction_detectors):
    total_waiting_time = 0.0
    total_stops = 0
    total_co2 = 0.0
    total_fuel = 0.0
    counted_vehicles = set()

    for det in junction_detectors:
        veh_ids = traci.lanearea.getLastStepVehicleIDs(det)
        for veh_id in veh_ids:
            if veh_id not in counted_vehicles:
                counted_vehicles.add(veh_id)
                try:
                    speed = traci.vehicle.getSpeed(veh_id)
                    wait_time = traci.vehicle.getWaitingTime(veh_id)
                    co2 = traci.vehicle.getCO2Emission(veh_id)
                    fuel = traci.vehicle.getFuelConsumption(veh_id)

                    total_waiting_time += wait_time
                    total_co2 += co2
                    total_fuel += fuel
                    if speed < 0.1:
                        total_stops += 1
                except:
                    continue
    return total_waiting_time, total_stops, total_co2, total_fuel

# Online Training Loop with Metrics
step_history = []
reward_kp_history = []
reward_dp_history = []
queue_kp_history = []
queue_dp_history = []

waiting_kp_history = []
waiting_dp_history = []
stops_kp_history = []
stops_dp_history = []
co2_kp_history = []
co2_dp_history = []
fuel_kp_history = []
fuel_dp_history = []

print("=== Starting Multi-Agent RL with Communication ===")
for step in range(TOTAL_STEPS):
    current_step = step

    joint_state = get_joint_state()

    action_kp = get_action("KP", joint_state, Q_table_KP)
    action_dp = get_action("DP", joint_state, Q_table_DP)

    apply_action("Kodrigktonos_Patision", action_kp, current_step)
    apply_action("Derigni_Patision", action_dp, current_step)

    traci.simulationStep()

    new_joint_state = get_joint_state()

    reward_kp = get_reward(new_joint_state[:6] + (new_joint_state[6],))
    reward_dp = get_reward(new_joint_state[7:-1] + (new_joint_state[-1],))

    update_q_table(joint_state, action_kp, reward_kp, new_joint_state, Q_table_KP)
    update_q_table(joint_state, action_dp, reward_dp, new_joint_state, Q_table_DP)

    if step % 100 == 0:

        step_history.append(step)
        reward_kp_history.append(reward_kp)
        reward_dp_history.append(reward_dp)
        queue_kp_history.append(sum(new_joint_state[:6]))
        queue_dp_history.append(sum(new_joint_state[7:-1]))

        wait_kp, stops_kp, co2_kp, fuel_kp = get_vehicle_metrics(detectors["Kodrigktonos_Patision"])
        wait_dp, stops_dp, co2_dp, fuel_dp = get_vehicle_metrics(detectors["Derigni_Patision"])

        waiting_kp_history.append(wait_kp)
        waiting_dp_history.append(wait_dp)
        stops_kp_history.append(stops_kp)
        stops_dp_history.append(stops_dp)
        co2_kp_history.append(co2_kp)
        co2_dp_history.append(co2_dp)
        fuel_kp_history.append(fuel_kp)
        fuel_dp_history.append(fuel_dp)

        print(
        f"\nStep {step} Metrics:"
        f"\n  [Kodrigktonos_Patision]  Reward: {reward_kp:.2f}  |  Queue: {sum(new_joint_state[:6])}  |  Waiting: {wait_kp:.2f}s"
        f"  |  Stops: {stops_kp}  |  CO₂: {co2_kp:.2f}g  |  Fuel: {fuel_kp:.2f}L"
        f"\n  [Derigni_Patision]       Reward: {reward_dp:.2f}  |  Queue: {sum(new_joint_state[7:-1])}  |  Waiting: {wait_dp:.2f}s"
        f"  |  Stops: {stops_dp}  |  CO₂: {co2_dp:.2f}g  |  Fuel: {fuel_dp:.2f}L"
        )


# Close SUMO 
traci.close()

print("\n========== Simulation Summary ==========")

print(f"Cumulative Reward (Kodrigktonos–Patision): {sum(reward_kp_history):.2f}")
print(f"Cumulative Reward (Derigni–Patision): {sum(reward_dp_history):.2f}")

print(f"Average Queue Length (Kodrigktonos–Patision): {sum(queue_kp_history) / len(queue_kp_history):.2f}")
print(f"Average Queue Length (Derigni–Patision): {sum(queue_dp_history) / len(queue_dp_history):.2f}")

print(f"Average Waiting Time (Kodrigktonos–Patision): {sum(waiting_kp_history) / len(waiting_kp_history):.2f} s")
print(f"Average Waiting Time (Derigni–Patision): {sum(waiting_dp_history) / len(waiting_dp_history):.2f} s")

print(f"Average Number of Stops (Kodrigktonos–Patision): {sum(stops_kp_history) / len(stops_kp_history):.2f}")
print(f"Average Number of Stops (Derigni–Patision): {sum(stops_dp_history) / len(stops_dp_history):.2f}")

print(f"Average CO2 Emissions (Kodrigktonos–Patision): {sum(co2_kp_history) / len(co2_kp_history):.2f} mg")
print(f"Average CO2 Emissions (Derigni–Patision): {sum(co2_dp_history) / len(co2_dp_history):.2f} mg")

print(f"Average Fuel Consumption (Kodrigktonos–Patision): {sum(fuel_kp_history) / len(fuel_kp_history):.2f} ml")
print(f"Average Fuel Consumption (Derigni–Patision): {sum(fuel_dp_history) / len(fuel_dp_history):.2f} ml")
print("========================================\n")


# Plotting
metrics = [
    (reward_kp_history, reward_dp_history, "Cumulative Reward per Junction", "Reward", ('blue', 'orange')),
    (queue_kp_history, queue_dp_history, "Queue Length per Junction", "Total Queue Length", ('blue', 'orange')),
    (waiting_kp_history, waiting_dp_history, "Average Waiting Time per Junction", "Waiting Time (s)", ('blue', 'orange')),
    (stops_kp_history, stops_dp_history, "Stops per Junction", "Number of Stops", ('blue', 'orange')),
    (co2_kp_history, co2_dp_history, "CO2 Emission per Junction", "CO2 Emission (mg)", ('blue', 'orange')),
    (fuel_kp_history, fuel_dp_history, "Fuel Consumption per Junction", "Fuel Consumption (ml)", ('blue', 'orange')),
]

plt.figure(figsize=(14, 10))

for i, (kp_data, dp_data, title, ylabel, (color_kp, color_dp)) in enumerate(metrics, 1):
    plt.subplot(3, 2, i)
    plt.plot(step_history, kp_data, color=color_kp, label='Kodrigktonos')
    plt.plot(step_history, dp_data, color=color_dp, label='Derigni')
    plt.xlabel("Simulation Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)

plt.suptitle("Multi Agent Traffic Lights With Communication Metrics Over Simulation Steps", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig(r"Athens Network\Case Kodrigktonos_Patision and Derigni_Patision\Results\Multi Agent Traffic Lights With Communication\Combined_Metrics.png", dpi=300, bbox_inches='tight')
plt.show()