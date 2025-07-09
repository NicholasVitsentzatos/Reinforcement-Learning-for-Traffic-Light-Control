import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import random

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
    '-c', 'Athens Network\Case Kodrigktonos_Patision\Data\osm.sumocfg',
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
traffic_light_id = "Kodrigktonos_Patision"

# PPO Model
class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        self.action_head = nn.Linear(64, output_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.action_head(x), self.value_head(x)

# PPO Agent
class PPOAgent:
    def __init__(self, input_dim, output_dim):
        self.policy = PolicyNet(input_dim, output_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.old_policy = PolicyNet(input_dim, output_dim)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state)
        logits, _ = self.old_policy(state_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, memory):
        states, actions, rewards, next_states, log_probs, dones = zip(*memory)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        old_log_probs = torch.stack(log_probs).detach()

        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            discounted_sum = reward + GAMMA * discounted_sum * (1 - done)
            returns.insert(0, discounted_sum)
        returns = torch.FloatTensor(returns)

        for _ in range(5):  # Epochs
            logits, values = agent.policy(states)
            dist = Categorical(logits=logits)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(actions.squeeze())

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
def get_state():
    dets = [f"Traffic Panel Detector {i+1}" for i in range(6)]
    queues = [traci.lanearea.getLastStepVehicleNumber(d) for d in dets]
    phase = traci.trafficlight.getPhase(traffic_light_id)
    return queues + [phase]

def get_reward(state):
    return -sum(state[:-1])

def apply_action(action, step):
    if action == 1 and step - last_switch_step[0] >= MIN_GREEN_STEPS:
        program = traci.trafficlight.getAllProgramLogics(traffic_light_id)[0]
        next_phase = (traci.trafficlight.getPhase(traffic_light_id) + 1) % len(program.phases)
        traci.trafficlight.setPhase(traffic_light_id, next_phase)
        last_switch_step[0] = step


# Agents & Memory
agent = PPOAgent(input_dim=7, output_dim=2)
memory = []
last_switch_step = [-MIN_GREEN_STEPS]
cumulative_reward = 0.0

# Metrics records
step_history, reward_history, queue_history = [], [], []
waiting_time_history, stops_history, co2_history, fuel_history = [], [], [], []
vehicle_stop_counts, vehicle_prev_stopped = {}, {}

# Main Loop
print("\n=== PPO Training ===")
for step in range(TOTAL_STEPS):
    current_state = get_state()
    action, log_prob = agent.select_action(current_state)
    apply_action(action, step)
    traci.simulationStep()

    next_state = get_state()
    reward = get_reward(next_state)
    done = False
    cumulative_reward += reward
    memory.append((current_state, action, reward, next_state, log_prob, done))

    # Stops tracking
    for v in traci.vehicle.getIDList():
        speed = traci.vehicle.getSpeed(v)
        stopped = speed < STOP_SPEED_THRESHOLD
        if v not in vehicle_stop_counts:
            vehicle_stop_counts[v] = 0
            vehicle_prev_stopped[v] = False
        if stopped and not vehicle_prev_stopped[v]:
            vehicle_stop_counts[v] += 1
        vehicle_prev_stopped[v] = stopped

    # PPO Update
    if (step + 1) % UPDATE_INTERVAL == 0:
        agent.update(memory)
        memory = []

    # Metric recording
    if step % 100 == 0:
        step_history.append(step)
        reward_history.append(cumulative_reward)
        queue_history.append(sum(next_state[:-1]))

        vehicles = traci.vehicle.getIDList()
        avg_wait = sum(traci.vehicle.getWaitingTime(v) for v in vehicles) / len(vehicles) if vehicles else 0.0
        total_stops = sum(vehicle_stop_counts.values())
        total_co2 = sum(traci.vehicle.getCO2Emission(v) for v in vehicles)
        total_fuel = sum(traci.vehicle.getFuelConsumption(v) for v in vehicles)

        waiting_time_history.append(avg_wait)
        stops_history.append(total_stops)
        co2_history.append(total_co2)
        fuel_history.append(total_fuel)

        print(f"[Step {step}] Reward: {reward:.2f}, Cum: {cumulative_reward:.2f}")
        print(f"Queue: {sum(next_state[:-1])}, Wait: {avg_wait:.2f}s, Stops: {total_stops}, CO₂: {total_co2:.2f}, Fuel: {total_fuel:.2f}")

# Close Simulation
traci.close()
print("=== PPO Training Finished ===")
print("\n========== Simulation Summary ==========")
print(f"Cumulative Reward: {reward_history[-1]:.2f}")
print(f"Average Queue Length: {sum(queue_history) / len(queue_history):.2f}")
print(f"Average Waiting Time: {sum(waiting_time_history) / len(waiting_time_history):.2f} s")
print(f"Average Number of Stops: {sum(stops_history) / len(stops_history):.2f}")
print(f"Average CO2 Emissions: {sum(co2_history) / len(co2_history):.2f} mg")
print(f"Average Fuel Consumption: {sum(fuel_history) / len(fuel_history):.2f} ml")
print("========================================\n")

# Plotting
metrics = [
    (reward_history, "Cumulative Reward", "Reward", 'blue'),
    (queue_history, "Total Queue Length", "Vehicles", 'orange'),
    (waiting_time_history, "Avg Waiting Time", "Seconds", 'green'),
    (stops_history, "Total Stops", "Stops", 'red'),
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

plt.suptitle("PPO Traffic Light Metrics Over Simulation Steps", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig(r"Athens Network\Case Kodrigktonos_Patision\Results\PPO Traffic Light\Combined_Metrics.png", dpi=300, bbox_inches='tight')
plt.show()