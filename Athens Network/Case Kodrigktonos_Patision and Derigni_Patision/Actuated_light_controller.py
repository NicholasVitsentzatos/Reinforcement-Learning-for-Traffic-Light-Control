import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Set SUMO_HOME and import TraCI
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

# SUMO config
Sumo_config = [
    'sumo-gui',
    '-c', 'Athens Network\Data\osm.sumocfg',
    '--step-length', '0.10',
    '--delay', '0',
    '--lateral-resolution', '0'
]

# PPO Classes
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_policy = nn.Linear(hidden_dim, action_dim)
        self.fc_value = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        policy_logits = self.fc_policy(x)
        value = self.fc_value(x)
        return policy_logits, value

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.memory = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        logits, _ = self.model(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob.item()

    def store_transition(self, transition):
        self.memory.append(transition)

    def train(self):
        if len(self.memory) == 0:
            return
        states = torch.FloatTensor([t[0] for t in self.memory])
        actions = torch.LongTensor([t[1] for t in self.memory]).unsqueeze(1)
        old_logprobs = torch.FloatTensor([t[2] for t in self.memory]).unsqueeze(1)
        rewards = [t[3] for t in self.memory]
        dones = [t[4] for t in self.memory]

        discounted_rewards = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0
            G = reward + self.gamma * G
            discounted_rewards.insert(0, G)
        discounted_rewards = torch.FloatTensor(discounted_rewards).unsqueeze(1)

        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        for _ in range(self.k_epochs):
            logits, state_values = self.model(states)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            new_logprobs = dist.log_prob(actions.squeeze()).unsqueeze(1)
            entropy = dist.entropy().mean()

            ratios = torch.exp(new_logprobs - old_logprobs)

            advantages = discounted_rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2).mean() + 0.5 * F.mse_loss(state_values, discounted_rewards) - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory = []

# Start SUMO
traci.start(Sumo_config)
traci.gui.setSchema("View #0", "real world")

# Simulation params and IDs
TOTAL_STEPS = 5000
STOP_SPEED_THRESHOLD = 0.1  # m/s

traffic_light_id_kodrigktonos = "Kodrigktonos_Patision"
traffic_light_id_derigni = "Derigni_Patision"

# Detector lists
detectors_kodrigktonos = [
    "Traffic Panel Detector 1", "Traffic Panel Detector 2", "Traffic Panel Detector 3",
    "Traffic Panel Detector 4", "Traffic Panel Detector 5", "Traffic Panel Detector 6"
]

detectors_derigni = [
    "Traffic Panel Detector 7", "Traffic Panel Detector 8", "Traffic Panel Detector 9",
    "Traffic Panel Detector 10", "Traffic Panel Detector 11"
]

# Data containers
step_history = []
reward_history = []
queue_history_kodrigktonos = []
queue_history_derigni = []
queue_history_total = []
waiting_time_history = []
stops_history = []
co2_history = []
fuel_history = []

vehicle_stop_counts = {}
vehicle_prev_stopped = {}

# Metric functions
def get_queue_length(detector_id):
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

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
    return -total_queue  # Simple negative queue length as reward

# Helper Functions
def get_state(detectors):
    return [get_queue_length(d) for d in detectors]

def apply_action(tls_id, action, prev_phase):
    if action == 1:
        # Get number of phases for a traffic light
        num_phases = len(traci.trafficlight.getAllProgramLogics(tls_id)[0].phases)

        # Calculate next phase index
        next_phase = (prev_phase + 1) % num_phases
        traci.trafficlight.setPhase(tls_id, next_phase)
        return next_phase
    else:
        traci.trafficlight.setPhase(tls_id, prev_phase)
        return prev_phase

# Initialize PPO Agents
state_dim_kodrigktonos = len(detectors_kodrigktonos)
state_dim_derigni = len(detectors_derigni)
action_dim = 2  # Keep or switch phase

agent_kodrigktonos = PPOAgent(state_dim_kodrigktonos, action_dim)
agent_derigni = PPOAgent(state_dim_derigni, action_dim)

prev_phase_kodrigktonos = traci.trafficlight.getPhase(traffic_light_id_kodrigktonos)
prev_phase_derigni = traci.trafficlight.getPhase(traffic_light_id_derigni)

cumulative_reward = 0.0

print("\n=== Starting PPO Actuated Traffic Light Simulation for Two Junctions ===")

# Main Simulation Loop
for step in range(TOTAL_STEPS):
    traci.simulationStep()

    # Vehicle stop tracking
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

    # Get states
    state_kodrigktonos = get_state(detectors_kodrigktonos)
    state_derigni = get_state(detectors_derigni)

    # Agents select actions
    action_kodrigktonos, logprob_kodrigktonos = agent_kodrigktonos.select_action(state_kodrigktonos)
    action_derigni, logprob_derigni = agent_derigni.select_action(state_derigni)

    # Apply actions
    current_phase_kodrigktonos = apply_action(traffic_light_id_kodrigktonos, action_kodrigktonos, prev_phase_kodrigktonos)
    current_phase_derigni = apply_action(traffic_light_id_derigni, action_derigni, prev_phase_derigni)

    # Calculate reward
    total_queue_kodrigktonos = sum(state_kodrigktonos)
    total_queue_derigni = sum(state_derigni)
    total_queue = total_queue_kodrigktonos + total_queue_derigni

    reward = get_reward(total_queue)
    cumulative_reward += reward

    # Store transitions
    done = (step == TOTAL_STEPS - 1)
    agent_kodrigktonos.store_transition((state_kodrigktonos, action_kodrigktonos, logprob_kodrigktonos, reward, done))
    agent_derigni.store_transition((state_derigni, action_derigni, logprob_derigni, reward, done))

    # Train PPO agents every 200 steps
    if step % 200 == 0 and step > 0:
        agent_kodrigktonos.train()
        agent_derigni.train()

    # Update previous phases
    prev_phase_kodrigktonos = current_phase_kodrigktonos
    prev_phase_derigni = current_phase_derigni

    # Calculate other metrics
    avg_wait = get_avg_waiting_time()
    total_stops = get_total_stops()
    co2 = get_total_co2()
    fuel = get_total_fuel()

    # Save and print metrics every 100 steps
    if step % 100 == 0:
        step_history.append(step)
        reward_history.append(cumulative_reward)
        queue_history_kodrigktonos.append(total_queue_kodrigktonos)
        queue_history_derigni.append(total_queue_derigni)
        queue_history_total.append(total_queue)
        waiting_time_history.append(avg_wait)
        stops_history.append(total_stops)
        co2_history.append(co2)
        fuel_history.append(fuel)

        print(f"Step {step} | Cumulative: {cumulative_reward:.2f} | Queue Kodrigktonos: {total_queue_kodrigktonos} | Queue Derigni: {total_queue_derigni} | "
              f"Total Queue: {total_queue} | Avg Wait: {avg_wait:.2f} s | Stops: {total_stops} | CO₂: {co2:.2f} mg | Fuel: {fuel:.2f} ml")

# Close SUMO
traci.close()
print("Simulation completed.")

print("\n========== Simulation Summary ==========")
print(f"Cumulative Reward: {reward_history[-1]:.2f}")

print(f"Average Queue Length (Kodrigktonos): {sum(queue_history_kodrigktonos) / len(queue_history_kodrigktonos):.2f}")
print(f"Average Queue Length (Derigni): {sum(queue_history_derigni) / len(queue_history_derigni):.2f}")
print(f"Average Queue Length (Total): {sum(queue_history_total) / len(queue_history_total):.2f}")

print(f"Average Waiting Time: {sum(waiting_time_history) / len(waiting_time_history):.2f} s")
print(f"Average Number of Stops: {sum(stops_history) / len(stops_history):.2f}")
print(f"Average CO2 Emissions: {sum(co2_history) / len(co2_history):.2f} mg")
print(f"Average Fuel Consumption: {sum(fuel_history) / len(fuel_history):.2f} ml")
print("========================================\n")


# Plotting
metrics = [
    (reward_history, "Cumulative Reward", "Cumulative Reward", 'blue'),
    (queue_history_kodrigktonos, "Queue Length Kodrigktonos", "Vehicles", 'orange'),
    (queue_history_derigni, "Queue Length Derigni", "Vehicles", 'green'),
    (queue_history_total, "Total Queue Length (Both Junctions)", "Vehicles", 'red'),
    (waiting_time_history, "Average Waiting Time", "Seconds", 'purple'),
    (stops_history, "Number of Stops", "Stops", 'brown'),
    (co2_history, "CO₂ Emissions", "mg", 'pink'),
    (fuel_history, "Fuel Consumption", "ml", 'gray'),
]

plt.figure(figsize=(14, 12))

for i, (history, title, ylabel, color) in enumerate(metrics, 1):
    plt.subplot(4, 2, i)
    plt.plot(step_history, history, color=color)
    plt.xlabel("Simulation Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

plt.suptitle("Actuated Traffic Lights Metrics Over Simulation Steps", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig(r"Athens Network\Case Kodrigktonos_Patision and Derigni_Patision\Results\Actuated Traffic Lights\Combined_Metrics.png", dpi=300, bbox_inches='tight')
plt.show()
