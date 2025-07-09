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
    '-c', 'Athens Network\Data\osm.sumocfg',
    '--step-length', '0.10',
    '--delay', '50',
    '--lateral-resolution', '0'
]

# Start SUMO
traci.start(Sumo_config)
traci.gui.setSchema("View #0", "real world")

# RL Parameters
TOTAL_STEPS = 5000
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
ACTIONS = [0, 1]  # 0: keep, 1: switch
Q_table = {}

MIN_GREEN_STEPS = 100
last_switch_KP = -MIN_GREEN_STEPS
last_switch_DP = -MIN_GREEN_STEPS
last_switch_CP = -MIN_GREEN_STEPS

# Detector ID lists
KP_detectors = [f"Traffic Panel Detector {i}" for i in range(1, 7)]
DP_detectors = [f"Traffic Panel Detector {i}" for i in range(7, 12)]
CP_detectors = [f"Traffic Panel Detector {i}" for i in range(12, 16)]

# Helper Functions 
def get_queue_length(detector_id):
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)

def get_state():
    # Junction 1: Kodrigktonos_Patision
    KP_EB = [get_queue_length(f"Traffic Panel Detector {i}") for i in range(1, 4)]
    KP_SB = [get_queue_length(f"Traffic Panel Detector {i}") for i in range(4, 7)]
    phase_KP = get_current_phase("Kodrigktonos_Patision")
    
    # Junction 2: Derigni_Patision
    DP_EB = [get_queue_length(f"Traffic Panel Detector {i}") for i in range(7, 10)]
    DP_SB = [get_queue_length(f"Traffic Panel Detector {i}") for i in range(10, 12)]
    phase_DP = get_current_phase("Derigni_Patision")
    
    # Junction 3: Cheven_Patision
    CP_EB = [get_queue_length(f"Traffic Panel Detector {i}") for i in range(12, 15)]
    CP_SB = [get_queue_length(f"Traffic Panel Detector {i}") for i in range(15, 16)]
    # Note: The detectors for Cheven_Patision are 12 to 15 (4 detectors),
    # Here, I assume the split: 12-14 as EB lanes, 15 as SB lane (adapt if needed)
    phase_CP = get_current_phase("Cheven_Patision")
    
    # Build full state tuple
    return tuple(KP_EB + KP_SB + [phase_KP] + DP_EB + DP_SB + [phase_DP] + CP_EB + CP_SB + [phase_CP])

def get_action_from_policy(state):
    # Number of combined actions: 2^3 = 8
    if state not in Q_table:
        Q_table[state] = np.zeros(len(ACTIONS)**3)
    if random.random() < EPSILON:
        return random.choice(range(len(ACTIONS)**3))
    return int(np.argmax(Q_table[state]))

def decode_actions(combined_action_index):
    # Decode combined index into individual junction actions
    a_kp = (combined_action_index // (len(ACTIONS)**2)) % len(ACTIONS)
    a_dp = (combined_action_index // len(ACTIONS)) % len(ACTIONS)
    a_cp = combined_action_index % len(ACTIONS)
    return a_kp, a_dp, a_cp

def apply_action(action, tls_id, current_step, last_switch_step):
    if action == 1 and current_step - last_switch_step >= MIN_GREEN_STEPS:
        logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        next_phase = (get_current_phase(tls_id) + 1) % len(logic.phases)
        traci.trafficlight.setPhase(tls_id, next_phase)
        return current_step
    return last_switch_step

def get_reward(state):
    # Sum queues for all junctions (excluding phases)
    return -float(sum(state[:-3]))

def update_Q_table(old_state, action_index, reward, new_state):
    if old_state not in Q_table:
        Q_table[old_state] = np.zeros(len(ACTIONS)**3)
    if new_state not in Q_table:
        Q_table[new_state] = np.zeros(len(ACTIONS)**3)
    old_q = Q_table[old_state][action_index]
    future_q = np.max(Q_table[new_state])
    Q_table[old_state][action_index] = old_q + ALPHA * (reward + GAMMA * future_q - old_q)

def get_metrics(detector_list):
    total_wait = 0.0
    total_stops = 0
    total_co2 = 0.0
    total_fuel = 0.0
    counted_vehicles = set()
    
    for det in detector_list:
        veh_ids = traci.lanearea.getLastStepVehicleIDs(det)
        for vid in veh_ids:
            if vid not in counted_vehicles:
                counted_vehicles.add(vid)
                try:
                    speed = traci.vehicle.getSpeed(vid)
                    total_wait += traci.vehicle.getWaitingTime(vid)
                    total_co2 += traci.vehicle.getCO2Emission(vid)
                    total_fuel += traci.vehicle.getFuelConsumption(vid)
                    if speed < 0.1:
                        total_stops += 1
                except:
                    continue
    return total_wait, total_stops, total_co2, total_fuel

# Main Training Loop 
step_history = []
reward_history = []
queue_history = []

# New metric histories for each junction
wait_kp_hist, wait_dp_hist, wait_cp_hist = [], [], []
stops_kp_hist, stops_dp_hist, stops_cp_hist = [], [], []
co2_kp_hist, co2_dp_hist, co2_cp_hist = [], [], []
fuel_kp_hist, fuel_dp_hist, fuel_cp_hist = [], [], []

cumulative_reward = 0.0

print("\n=== Starting RL Agent for Three Junctions ===")
for step in range(TOTAL_STEPS):
    current_simulation_step = step
    state = get_state()
    
    action_index = get_action_from_policy(state)
    action_kp, action_dp, action_cp = decode_actions(action_index)
    
    last_switch_KP = apply_action(action_kp, "Kodrigktonos_Patision", current_simulation_step, last_switch_KP)
    last_switch_DP = apply_action(action_dp, "Derigni_Patision", current_simulation_step, last_switch_DP)
    last_switch_CP = apply_action(action_cp, "Cheven_Patision", current_simulation_step, last_switch_CP)

    traci.simulationStep()
    
    new_state = get_state()
    reward = get_reward(new_state)
    cumulative_reward += reward

    update_Q_table(state, action_index, reward, new_state)

    # Metrics
    wait_kp, stops_kp, co2_kp, fuel_kp = get_metrics(KP_detectors)
    wait_dp, stops_dp, co2_dp, fuel_dp = get_metrics(DP_detectors)
    wait_cp, stops_cp, co2_cp, fuel_cp = get_metrics(CP_detectors)

    if step % 100 == 0:

        step_history.append(step)
        reward_history.append(cumulative_reward)
        queue_history.append(sum(new_state[:-3]))

        wait_kp_hist.append(wait_kp)
        wait_dp_hist.append(wait_dp)
        wait_cp_hist.append(wait_cp)

        stops_kp_hist.append(stops_kp)
        stops_dp_hist.append(stops_dp)
        stops_cp_hist.append(stops_cp)

        co2_kp_hist.append(co2_kp)
        co2_dp_hist.append(co2_dp)
        co2_cp_hist.append(co2_cp)

        fuel_kp_hist.append(fuel_kp)
        fuel_dp_hist.append(fuel_dp)
        fuel_cp_hist.append(fuel_cp)

        print(
                f"\nStep {step} Metrics:"
                f"\n  Cumulative Reward: {cumulative_reward:.2f}  |  Total Queue: {queue_history[-1]}"
                f"\n  [Kodrigktonos_Patision]  Waiting: {wait_kp:.2f}s  |  Stops: {stops_kp}  |  CO₂: {co2_kp:.2f}g  |  Fuel: {fuel_kp:.2f}L"
                f"\n  [Derigni_Patision]       Waiting: {wait_dp:.2f}s  |  Stops: {stops_dp}  |  CO₂: {co2_dp:.2f}g  |  Fuel: {fuel_dp:.2f}L"
                f"\n  [Cheven_Patision]        Waiting: {wait_cp:.2f}s  |  Stops: {stops_cp}  |  CO₂: {co2_cp:.2f}g  |  Fuel: {fuel_cp:.2f}L"
            )
# Close SUMO
traci.close()

print("\n====== Simulation Summary ======")
print(f"Cumulative Reward: {reward_history[-1]:.2f}")

print(f"Average Queue Length: {sum(queue_history) / len(queue_history):.2f}")

print(f"Average Waiting Time (Kodrigktonos): {sum(wait_kp_hist) / len(wait_kp_hist):.2f} s")
print(f"Average Waiting Time (Derigni): {sum(wait_dp_hist) / len(wait_dp_hist):.2f} s")
print(f"Average Waiting Time (Cheven): {sum(wait_cp_hist) / len(wait_cp_hist):.2f} s")

print(f"Average Stops (Kodrigktonos): {sum(stops_kp_hist) / len(stops_kp_hist):.2f}")
print(f"Average Stops (Derigni): {sum(stops_dp_hist) / len(stops_dp_hist):.2f}")
print(f"Average Stops (Cheven): {sum(stops_cp_hist) / len(stops_cp_hist):.2f}")

print(f"Average CO2 Emissions (Kodrigktonos): {sum(co2_kp_hist) / len(co2_kp_hist):.2f} mg")
print(f"Average CO2 Emissions (Derigni): {sum(co2_dp_hist) / len(co2_dp_hist):.2f} mg")
print(f"Average CO2 Emissions (Cheven): {sum(co2_cp_hist) / len(co2_cp_hist):.2f} mg")

print(f"Average Fuel Consumption (Kodrigktonos): {sum(fuel_kp_hist) / len(fuel_kp_hist):.2f} ml")
print(f"Average Fuel Consumption (Derigni): {sum(fuel_dp_hist) / len(fuel_dp_hist):.2f} ml")
print(f"Average Fuel Consumption (Cheven): {sum(fuel_cp_hist) / len(fuel_cp_hist):.2f} ml")

print("=================================\n")


# print("\nTraining completed. Final Q-table size:", len(Q_table))
# for st, qvals in Q_table.items():
#     print(f"State: {st} -> Q-values: {qvals}")

# Plotting 
plt.figure(figsize=(14, 12))

plt.subplot(3, 2, 1)
plt.plot(step_history, reward_history, label="Cumulative Reward", color='blue')
plt.xlabel("Step")
plt.ylabel("Cumulative Reward")
plt.title("Three-Junction RL Training: Cumulative Reward")
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(step_history, queue_history, label="Total Queue Length", color='orange')
plt.xlabel("Step")
plt.ylabel("Queue Length")
plt.title("Three-Junction RL Training: Queue Length")
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 3)
plt.plot(step_history, wait_kp_hist, label="Kodrigktonos")
plt.plot(step_history, wait_dp_hist, label="Derigni")
plt.plot(step_history, wait_cp_hist, label="Cheven")
plt.xlabel("Step")
plt.ylabel("Waiting Time (s)")
plt.title("Average Waiting Time per Junction")
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(step_history, stops_kp_hist, label="Kodrigktonos")
plt.plot(step_history, stops_dp_hist, label="Derigni")
plt.plot(step_history, stops_cp_hist, label="Cheven")
plt.xlabel("Step")
plt.ylabel("Stops")
plt.title("Number of Stops per Junction")
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 5)
plt.plot(step_history, co2_kp_hist, label="Kodrigktonos")
plt.plot(step_history, co2_dp_hist, label="Derigni")
plt.plot(step_history, co2_cp_hist, label="Cheven")
plt.xlabel("Step")
plt.ylabel("CO₂ (mg)")
plt.title("CO₂ Emissions per Junction")
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 6)
plt.plot(step_history, fuel_kp_hist, label="Kodrigktonos")
plt.plot(step_history, fuel_dp_hist, label="Derigni")
plt.plot(step_history, fuel_cp_hist, label="Cheven")
plt.xlabel("Step")
plt.ylabel("Fuel (ml)")
plt.title("Fuel Consumption per Junction")
plt.grid(True)
plt.legend()


plt.suptitle("Q Traffic Lights Metrics Over Simulation Steps", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig(r"Athens Network\Case Kodrigktonos_Patision Derigni_Patision and Chevden_Patision\Results\Q Traffic Lights\Combined_Metrics.png", dpi=300, bbox_inches='tight')
plt.show()