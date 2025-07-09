import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# SUMO Setup
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

# SUMO Config
sumo_config = [
    'sumo-gui',
    '-c', 'Athens Network\Data\osm.sumocfg',
    '--step-length', '0.10',
    '--delay', '50',
    '--lateral-resolution', '0'
]
traci.start(sumo_config)
traci.gui.setSchema("View #0", "real world")

# PPO Constants
GAMMA = 0.99
EPS_CLIP = 0.2
LR = 0.0003
UPDATE_INTERVAL = 200
TOTAL_STEPS = 5000
MIN_GREEN_STEPS = 100
STOP_SPEED_THRESHOLD = 0.1

# Detector Setup
KP_detectors = [f"Traffic Panel Detector {i}" for i in range(1, 7)]
DP_detectors = [f"Traffic Panel Detector {i}" for i in range(7, 12)]

# Traffic Light IDs
TLS_KP = "Kodrigktonos_Patision"
TLS_DP = "Derigni_Patision"

# PPO Model
class PolicyNet(nn.Module):
    def __init__(self, input_dim):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        self.action_head = nn.Linear(64, 2)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.action_head(x), self.value_head(x)

# PPO Agent
class PPOAgent:
    def __init__(self, input_dim):
        self.policy = PolicyNet(input_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.old_policy = PolicyNet(input_dim)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state)
        logits, _ = self.old_policy(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, memory):
        states, actions, rewards, next_states, log_probs, dones = zip(*memory)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        old_log_probs = torch.stack(log_probs).detach()

        returns = []
        discounted = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            discounted = reward + GAMMA * discounted * (1 - done)
            returns.insert(0, discounted)
        returns = torch.FloatTensor(returns)

        for _ in range(4):  # PPO Epochs
            logits, values = self.policy(states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions.squeeze())
            entropy = dist.entropy().mean()

            ratios = torch.exp(new_log_probs - old_log_probs)
            advantages = returns - values.squeeze().detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages

            loss = -torch.min(surr1, surr2).mean() + 0.5 * self.MseLoss(values.squeeze(), returns) - 0.01 * entropy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.old_policy.load_state_dict(self.policy.state_dict())

# Helper Functions
def get_queue_length(det_id):
    return traci.lanearea.getLastStepVehicleNumber(det_id)

def get_state(detectors, tls_id):
    q = [get_queue_length(d) for d in detectors]
    phase = traci.trafficlight.getPhase(tls_id)
    return q + [phase]

def get_reward(state):
    return -sum(state[:-1])

def apply_action(action, tls_id, current_step, last_switch_step):
    if action == 1 and current_step - last_switch_step >= MIN_GREEN_STEPS:
        logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        next_phase = (traci.trafficlight.getPhase(tls_id) + 1) % len(logic.phases)
        traci.trafficlight.setPhase(tls_id, next_phase)
        return current_step
    return last_switch_step

def get_metrics(detector_list):
    total_wait = total_stops = total_co2 = total_fuel = 0.0
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
                    if speed < STOP_SPEED_THRESHOLD:
                        total_stops += 1
                except:
                    continue
    return total_wait, total_stops, total_co2, total_fuel

# Agents & Memory
agent_kp = PPOAgent(7)
agent_dp = PPOAgent(6)

memory_kp, memory_dp = [], []
last_switch_KP, last_switch_DP = -MIN_GREEN_STEPS, -MIN_GREEN_STEPS
cumulative_reward = 0.0

# Metrics records
step_history, reward_history, queue_history = [], [], []
wait_kp_hist, wait_dp_hist = [], []
stops_kp_hist, stops_dp_hist = [], []
co2_kp_hist, co2_dp_hist = [], []
fuel_kp_hist, fuel_dp_hist = [], []

# Main Loop
print("\n=== PPO Training for 2 Junctions ===")
for step in range(TOTAL_STEPS):
    s_kp = get_state(KP_detectors, TLS_KP)
    s_dp = get_state(DP_detectors, TLS_DP)

    a_kp, logp_kp = agent_kp.select_action(s_kp)
    a_dp, logp_dp = agent_dp.select_action(s_dp)

    last_switch_KP = apply_action(a_kp, TLS_KP, step, last_switch_KP)
    last_switch_DP = apply_action(a_dp, TLS_DP, step, last_switch_DP)

    traci.simulationStep()

    s_kp_new = get_state(KP_detectors, TLS_KP)
    s_dp_new = get_state(DP_detectors, TLS_DP)

    r_kp = get_reward(s_kp_new)
    r_dp = get_reward(s_dp_new)
    cumulative_reward += (r_kp + r_dp)

    memory_kp.append((s_kp, a_kp, r_kp, s_kp_new, logp_kp, False))
    memory_dp.append((s_dp, a_dp, r_dp, s_dp_new, logp_dp, False))

    if (step + 1) % UPDATE_INTERVAL == 0:
        agent_kp.update(memory_kp)
        agent_dp.update(memory_dp)
        memory_kp.clear()
        memory_dp.clear()

    # Metrics
    wait_kp, stops_kp, co2_kp, fuel_kp = get_metrics(KP_detectors)
    wait_dp, stops_dp, co2_dp, fuel_dp = get_metrics(DP_detectors)

    if step % 100 == 0:
        step_history.append(step)
        reward_history.append(cumulative_reward)
        queue_history.append(sum(s_kp_new[:-1]) + sum(s_dp_new[:-1]))

        wait_kp_hist.append(wait_kp)
        wait_dp_hist.append(wait_dp)
        stops_kp_hist.append(stops_kp)
        stops_dp_hist.append(stops_dp)
        co2_kp_hist.append(co2_kp)
        co2_dp_hist.append(co2_dp)
        fuel_kp_hist.append(fuel_kp)
        fuel_dp_hist.append(fuel_dp)

        print(
        f"\nStep {step} Metrics:"
        f"\n  Cumulative Reward: {cumulative_reward:.2f}  |  Total Queue: {sum(s_kp_new[:-1]) + sum(s_dp_new[:-1])}"
        f"\n  [Kodrigktonos_Patision]  Waiting: {wait_kp:.2f}s  |  Stops: {stops_kp}  |  CO₂: {co2_kp:.2f}g  |  Fuel: {fuel_kp:.2f}L"
        f"\n  [Derigni_Patision]       Waiting: {wait_dp:.2f}s  |  Stops: {stops_dp}  |  CO₂: {co2_dp:.2f}g  |  Fuel: {fuel_dp:.2f}L"
        )

traci.close()

print("\n========== Simulation Summary ==========")
print(f"Cumulative Reward: {reward_history[-1]:.2f}")
print(f"Average Total Queue Length: {sum(queue_history) / len(queue_history):.2f}")

print(f"Average Waiting Time (Kodrigktonos–Patision): {sum(wait_kp_hist) / len(wait_kp_hist):.2f} s")
print(f"Average Waiting Time (Derigni–Patision): {sum(wait_dp_hist) / len(wait_dp_hist):.2f} s")

print(f"Average Number of Stops (Kodrigktonos–Patision): {sum(stops_kp_hist) / len(stops_kp_hist):.2f}")
print(f"Average Number of Stops (Derigni–Patision): {sum(stops_dp_hist) / len(stops_dp_hist):.2f}")

print(f"Average CO2 Emissions (Kodrigktonos–Patision): {sum(co2_kp_hist) / len(co2_kp_hist):.2f} mg")
print(f"Average CO2 Emissions (Derigni–Patision): {sum(co2_dp_hist) / len(co2_dp_hist):.2f} mg")

print(f"Average Fuel Consumption (Kodrigktonos–Patision): {sum(fuel_kp_hist) / len(fuel_kp_hist):.2f} ml")
print(f"Average Fuel Consumption (Derigni–Patision): {sum(fuel_dp_hist) / len(fuel_dp_hist):.2f} ml")
print("========================================\n")


# Plotting
plt.figure(figsize=(16, 12))

# 1. Cumulative Reward (single line)
plt.subplot(3, 2, 1)
plt.plot(step_history, reward_history, color='blue')
plt.title("Cumulative Reward")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.grid(True)

# 2. Total Queue Length (single line)
plt.subplot(3, 2, 2)
plt.plot(step_history, queue_history, color='orange')
plt.title("Total Queue Length")
plt.xlabel("Step")
plt.ylabel("Queue Length")
plt.grid(True)

# 3. Average Waiting Time (two lines)
plt.subplot(3, 2, 3)
plt.plot(step_history, wait_kp_hist, label="Kodrigktonos", color='blue')
plt.plot(step_history, wait_dp_hist, label="Derigni", color='orange')
plt.title("Average Waiting Time")
plt.xlabel("Step")
plt.ylabel("Seconds")
plt.legend()
plt.grid(True)

# 4. Number of Stops (two lines)
plt.subplot(3, 2, 4)
plt.plot(step_history, stops_kp_hist, label="Kodrigktonos", color='blue')
plt.plot(step_history, stops_dp_hist, label="Derigni", color='orange')
plt.title("Number of Stops")
plt.xlabel("Step")
plt.ylabel("Stops")
plt.legend()
plt.grid(True)

# 5. CO2 Emissions (two lines)
plt.subplot(3, 2, 5)
plt.plot(step_history, co2_kp_hist, label="Kodrigktonos", color='blue')
plt.plot(step_history, co2_dp_hist, label="Derigni", color='orange')
plt.title("CO2 Emissions")
plt.xlabel("Step")
plt.ylabel("mg")
plt.legend()
plt.grid(True)

# 6. Fuel Consumption (two lines)
plt.subplot(3, 2, 6)
plt.plot(step_history, fuel_kp_hist, label="Kodrigktonos", color='blue')
plt.plot(step_history, fuel_dp_hist, label="Derigni", color='orange')
plt.title("Fuel Consumption")
plt.xlabel("Step")
plt.ylabel("ml")
plt.legend()
plt.grid(True)

plt.suptitle("PPO Traffic Lights Metrics Over Simulation Steps", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig(r"Athens Network\Case Kodrigktonos_Patision and Derigni_Patision\Results\PPO Traffic Lights\Combined_Metrics.png", dpi=300, bbox_inches='tight')
plt.show()