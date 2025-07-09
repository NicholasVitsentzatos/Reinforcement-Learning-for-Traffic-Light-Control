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

# Q-Tables and last switch records
Q_table_KP = {}
Q_table_DP = {}
Q_table_CP = {}

last_switch_step = {
    "Kodrigktonos_Patision": -MIN_GREEN_STEPS,
    "Derigni_Patision": -MIN_GREEN_STEPS,
    "Cheven_Patision": -MIN_GREEN_STEPS
}

# Detector ID mappings
detectors = {
    "Kodrigktonos_Patision": [
        "Traffic Panel Detector 1", "Traffic Panel Detector 2", "Traffic Panel Detector 3",
        "Traffic Panel Detector 4", "Traffic Panel Detector 5", "Traffic Panel Detector 6"
    ],
    "Derigni_Patision": [
        "Traffic Panel Detector 7", "Traffic Panel Detector 8",
        "Traffic Panel Detector 9", "Traffic Panel Detector 10", "Traffic Panel Detector 11"
    ],
    "Cheven_Patision": [
        "Traffic Panel Detector 12", "Traffic Panel Detector 13",
        "Traffic Panel Detector 14", "Traffic Panel Detector 15"
    ]
}

# FUNCTIONS 
def get_queue_length(detector_id):
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)

def get_state(tls_id):
    detector_ids = detectors[tls_id]
    queue_lengths = [get_queue_length(det_id) for det_id in detector_ids]
    current_phase = get_current_phase(tls_id)
    return tuple(queue_lengths + [current_phase])

# New function to get joint state for communication
def get_joint_state():
    state_kp = get_state("Kodrigktonos_Patision")
    state_dp = get_state("Derigni_Patision")
    state_cp = get_state("Cheven_Patision")
    # Combine queue lengths and phases from all junctions
    joint_state = state_kp[:-1] + (state_kp[-1],) + \
                  state_dp[:-1] + (state_dp[-1],) + \
                  state_cp[:-1] + (state_cp[-1],)
    return joint_state

def get_reward(state, junction_prefix):
    # Extract the queue lengths relevant for the junction from the joint state
    if junction_prefix == "KP":
        queues = state[:len(detectors["Kodrigktonos_Patision"])]
    elif junction_prefix == "DP":
        start = len(detectors["Kodrigktonos_Patision"]) + 1
        end = start + len(detectors["Derigni_Patision"])
        queues = state[start:end]
    else:  # CP
        start = len(detectors["Kodrigktonos_Patision"]) + 1 + len(detectors["Derigni_Patision"]) + 1
        end = start + len(detectors["Cheven_Patision"])
        queues = state[start:end]
    return -float(sum(queues))

def get_action(state, q_table):
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
reward_cp_history = []

queue_kp_history = []
queue_dp_history = []
queue_cp_history = []

waiting_kp_history = []
waiting_dp_history = []
waiting_cp_history = []

stops_kp_history = []
stops_dp_history = []
stops_cp_history = []

co2_kp_history = []
co2_dp_history = []
co2_cp_history = []

fuel_kp_history = []
fuel_dp_history = []
fuel_cp_history = []

print("=== Starting Tri-Agent RL Training with Communication ===")
for step in range(TOTAL_STEPS):
    current_step = step

    joint_state = get_joint_state()

    # Agent KP
    action_kp = get_action(joint_state, Q_table_KP)
    apply_action("Kodrigktonos_Patision", action_kp, current_step)

    # Agent DP
    action_dp = get_action(joint_state, Q_table_DP)
    apply_action("Derigni_Patision", action_dp, current_step)

    # Agent CP
    action_cp = get_action(joint_state, Q_table_CP)
    apply_action("Cheven_Patision", action_cp, current_step)

    traci.simulationStep()

    new_joint_state = get_joint_state()

    reward_kp = get_reward(new_joint_state, "KP")
    reward_dp = get_reward(new_joint_state, "DP")
    reward_cp = get_reward(new_joint_state, "CP")

    update_q_table(joint_state, action_kp, reward_kp, new_joint_state, Q_table_KP)
    update_q_table(joint_state, action_dp, reward_dp, new_joint_state, Q_table_DP)
    update_q_table(joint_state, action_cp, reward_cp, new_joint_state, Q_table_CP)

    if step % 100 == 0:

        step_history.append(step)

        reward_kp_history.append(reward_kp)
        reward_dp_history.append(reward_dp)
        reward_cp_history.append(reward_cp)

        queue_kp_history.append(sum(new_joint_state[:len(detectors["Kodrigktonos_Patision"])]))
        start_dp = len(detectors["Kodrigktonos_Patision"]) + 1
        end_dp = start_dp + len(detectors["Derigni_Patision"])
        queue_dp_history.append(sum(new_joint_state[start_dp:end_dp]))
        start_cp = end_dp + 1
        end_cp = start_cp + len(detectors["Cheven_Patision"])
        queue_cp_history.append(sum(new_joint_state[start_cp:end_cp]))

        wait_kp, stops_kp, co2_kp, fuel_kp = get_vehicle_metrics(detectors["Kodrigktonos_Patision"])
        wait_dp, stops_dp, co2_dp, fuel_dp = get_vehicle_metrics(detectors["Derigni_Patision"])
        wait_cp, stops_cp, co2_cp, fuel_cp = get_vehicle_metrics(detectors["Cheven_Patision"])

        waiting_kp_history.append(wait_kp)
        waiting_dp_history.append(wait_dp)
        waiting_cp_history.append(wait_cp)

        stops_kp_history.append(stops_kp)
        stops_dp_history.append(stops_dp)
        stops_cp_history.append(stops_cp)

        co2_kp_history.append(co2_kp)
        co2_dp_history.append(co2_dp)
        co2_cp_history.append(co2_cp)

        fuel_kp_history.append(fuel_kp)
        fuel_dp_history.append(fuel_dp)
        fuel_cp_history.append(fuel_cp)

        print(
            f"\nStep {step} Metrics:"
            f"\n  [Kodrigktonos_Patision]  Reward: {reward_kp:.2f}  |  Queue: {sum(new_joint_state[:len(detectors['Kodrigktonos_Patision'])])}  |  Waiting: {wait_kp:.2f}s"
            f"  |  Stops: {stops_kp}  |  CO₂: {co2_kp:.2f}g  |  Fuel: {fuel_kp:.2f}L"
            f"\n  [Derigni_Patision]       Reward: {reward_dp:.2f}  |  Queue: {sum(new_joint_state[start_dp:end_dp])}  |  Waiting: {wait_dp:.2f}s"
            f"  |  Stops: {stops_dp}  |  CO₂: {co2_dp:.2f}g  |  Fuel: {fuel_dp:.2f}L"
            f"\n  [Cheven_Patision]        Reward: {reward_cp:.2f}  |  Queue: {sum(new_joint_state[start_cp:end_cp])}  |  Waiting: {wait_cp:.2f}s"
            f"  |  Stops: {stops_cp}  |  CO₂: {co2_cp:.2f}g  |  Fuel: {fuel_cp:.2f}L"
        )

# Close SUMO 
traci.close()

print("\n========== Simulation Summary ==========")

print(f"Cumulative Reward (Kodrigktonos): {sum(reward_kp_history):.2f}")
print(f"Cumulative Reward (Derigni): {sum(reward_dp_history):.2f}")
print(f"Cumulative Reward (Cheven): {sum(reward_cp_history):.2f}")

print(f"Average Queue Length (Kodrigktonos): {sum(queue_kp_history) / len(queue_kp_history):.2f}")
print(f"Average Queue Length (Derigni): {sum(queue_dp_history) / len(queue_dp_history):.2f}")
print(f"Average Queue Length (Cheven): {sum(queue_cp_history) / len(queue_cp_history):.2f}")

print(f"Average Waiting Time (Kodrigktonos): {sum(waiting_kp_history) / len(waiting_kp_history):.2f} s")
print(f"Average Waiting Time (Derigni): {sum(waiting_dp_history) / len(waiting_dp_history):.2f} s")
print(f"Average Waiting Time (Cheven): {sum(waiting_cp_history) / len(waiting_cp_history):.2f} s")

print(f"Average Number of Stops (Kodrigktonos): {sum(stops_kp_history) / len(stops_kp_history):.2f}")
print(f"Average Number of Stops (Derigni): {sum(stops_dp_history) / len(stops_dp_history):.2f}")
print(f"Average Number of Stops (Cheven): {sum(stops_cp_history) / len(stops_cp_history):.2f}")

print(f"Average CO2 Emissions (Kodrigktonos): {sum(co2_kp_history) / len(co2_kp_history):.2f} mg")
print(f"Average CO2 Emissions (Derigni): {sum(co2_dp_history) / len(co2_dp_history):.2f} mg")
print(f"Average CO2 Emissions (Cheven): {sum(co2_cp_history) / len(co2_cp_history):.2f} mg")

print(f"Average Fuel Consumption (Kodrigktonos): {sum(fuel_kp_history) / len(fuel_kp_history):.2f} ml")
print(f"Average Fuel Consumption (Derigni): {sum(fuel_dp_history) / len(fuel_dp_history):.2f} ml")
print(f"Average Fuel Consumption (Cheven): {sum(fuel_cp_history) / len(fuel_cp_history):.2f} ml")

print("========================================\n")


# Plotting
plt.figure(figsize=(14, 12))

plt.subplot(3, 2, 1)
plt.plot(step_history, reward_kp_history, label="Kodrigktonos")
plt.plot(step_history, reward_dp_history, label="Derigni")
plt.plot(step_history, reward_cp_history, label="Cheven")
plt.xlabel("Simulation Step")
plt.ylabel("Reward")
plt.title("Cumulative Reward per Junction")
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(step_history, queue_kp_history, label="Kodrigktonos")
plt.plot(step_history, queue_dp_history, label="Derigni")
plt.plot(step_history, queue_cp_history, label="Cheven")
plt.xlabel("Simulation Step")
plt.ylabel("Queue Length")
plt.title("Queue Length per Junction")
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 3)
plt.plot(step_history, waiting_kp_history, label="Kodrigktonos")
plt.plot(step_history, waiting_dp_history, label="Derigni")
plt.plot(step_history, waiting_cp_history, label="Cheven")
plt.xlabel("Simulation Step")
plt.ylabel("Waiting Time (s)")
plt.title("Average Waiting Time per Junction")
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(step_history, stops_kp_history, label="Kodrigktonos")
plt.plot(step_history, stops_dp_history, label="Derigni")
plt.plot(step_history, stops_cp_history, label="Cheven")
plt.xlabel("Simulation Step")
plt.ylabel("Stops")
plt.title("Number of Stops per Junction")
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 5)
plt.plot(step_history, co2_kp_history, label="Kodrigktonos")
plt.plot(step_history, co2_dp_history, label="Derigni")
plt.plot(step_history, co2_cp_history, label="Cheven")
plt.xlabel("Simulation Step")
plt.ylabel("CO₂ Emission (mg)")
plt.title("CO₂ Emission per Junction")
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(step_history, fuel_kp_history, label="Kodrigktonos")
plt.plot(step_history, fuel_dp_history, label="Derigni")
plt.plot(step_history, fuel_cp_history, label="Cheven")
plt.xlabel("Simulation Step")
plt.ylabel("Fuel Consumption (ml)")
plt.title("Fuel Consumption per Junction")
plt.legend()
plt.grid(True)

plt.suptitle("Multi Agent With Communication Metrics Over Simulation Steps", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig(r"Athens Network\Case Kodrigktonos_Patision Derigni_Patision and Chevden_Patision\Results\Multi Agent Traffic Lights With Communication\Combined_Metrics.png", dpi=300, bbox_inches='tight')
plt.show()
