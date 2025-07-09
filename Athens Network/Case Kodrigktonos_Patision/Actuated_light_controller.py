import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Set SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

# Define SUMO config
sumo_config = [
    "sumo-gui",
    "-c", "Athens Network\Case Kodrigktonos_Patision\Data\osm.sumocfg",
    "--step-length", "0.1",
    "--delay", "0",
    "--lateral-resolution", "0"
]

# PPO Agent
class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super(PPOAgent, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def act(self, state):
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate(self, state, action):
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(state).squeeze()
        return log_probs, value, entropy

class Memory:
    def __init__(self):
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones = [], []

    def clear(self):
        self.__init__()

# Constants and setup
TOTAL_STEPS = 5000
GAMMA = 0.99
EPS_CLIP = 0.2
PPO_EPOCHS = 4
BATCH_SIZE = 32
UPDATE_EVERY = 200
MIN_GREEN_TIME = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment and agent
tls_id = "Kodrigktonos_Patision"
detectors = [f"Traffic Panel Detector {i}" for i in range(1, 7)]
STOP_SPEED_THRESHOLD = 0.1
agent = PPOAgent(state_dim=6, action_dim=2).to(device)
optimizer = optim.Adam(agent.parameters(), lr=0.0003)
memory = Memory()

# Metrics
step_history, reward_history, queue_history = [], [], []
waiting_time_history, stops_history = [], []
co2_history, fuel_history = [], []
vehicle_stop_counts = {}
vehicle_prev_stopped = {}
last_switch_step = -MIN_GREEN_TIME

# Helper functions
def get_state():
    return torch.tensor([traci.lanearea.getLastStepVehicleNumber(d) for d in detectors], dtype=torch.float32).to(device)

def get_reward():
    return -sum(traci.lanearea.getLastStepVehicleNumber(d) for d in detectors)

def apply_action(action, current_step, last_switch):
    if action == 1 and current_step - last_switch >= MIN_GREEN_TIME:
        logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        next_phase = (traci.trafficlight.getPhase(tls_id) + 1) % len(logic.phases)
        traci.trafficlight.setPhase(tls_id, next_phase)
        return current_step
    return last_switch

def get_avg_waiting_time():
    vehicles = traci.vehicle.getIDList()
    return np.mean([traci.vehicle.getWaitingTime(v) for v in vehicles]) if vehicles else 0.0

def get_total_stops():
    return sum(vehicle_stop_counts.values())

def get_total_co2():
    return sum(traci.vehicle.getCO2Emission(v) for v in traci.vehicle.getIDList())

def get_total_fuel():
    return sum(traci.vehicle.getFuelConsumption(v) for v in traci.vehicle.getIDList())

# Start SUMO
traci.start(sumo_config)
traci.gui.setSchema("View #0", "real world")


# Simulation loop
print("\n=== Starting PPO Actuated Traffic Light Simulation ===")
cumulative_reward = 0
log_probs_buffer, values_buffer, rewards_buffer, dones_buffer = [], [], [], []

for step in range(TOTAL_STEPS):
    current_simulation_step = step

    # Track stops per vehicle
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

    # Get state and decide action
    state = get_state()
    action, log_prob = agent.act(state)
    last_switch_step = apply_action(action, current_simulation_step, last_switch_step)

    # Advance simulation
    traci.simulationStep()

    # Get reward and store experience
    reward = get_reward()
    cumulative_reward += reward
    done = False  # No terminal condition in SUMO

    memory.states.append(state.cpu())
    memory.actions.append(torch.tensor(action))
    memory.log_probs.append(log_prob.cpu())
    memory.rewards.append(reward)
    memory.dones.append(done)

    # PPO update every few steps
    if step > 0 and step % UPDATE_EVERY == 0:
        # Convert memory to tensors
        states = torch.stack(memory.states).to(device)
        actions = torch.tensor(memory.actions).to(device)
        log_probs_old = torch.stack(memory.log_probs).detach().to(device)  # Detach here
        returns = []
        G = 0
        for reward in reversed(memory.rewards):
            G = reward + GAMMA * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        for _ in range(PPO_EPOCHS):
            # Recompute new log_probs and values each epoch to build fresh computation graph
            log_probs_new, values, entropy = agent.evaluate(states, actions)

            advantages = returns - values.detach()
            ratio = torch.exp(log_probs_new - log_probs_old)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, returns)
            entropy_loss = -0.01 * entropy.mean()
            loss = actor_loss + 0.5 * critic_loss + entropy_loss

            optimizer.zero_grad()
            loss.backward()  # Safe now
            optimizer.step()

        memory.clear()

    # Record every 100 steps
    if step % 100 == 0:
        queue_len = sum(traci.lanearea.getLastStepVehicleNumber(d) for d in detectors)
        avg_wait = get_avg_waiting_time()
        stops = get_total_stops()
        co2 = get_total_co2()
        fuel = get_total_fuel()

        step_history.append(step)
        reward_history.append(cumulative_reward)
        queue_history.append(queue_len)
        waiting_time_history.append(avg_wait)
        stops_history.append(stops)
        co2_history.append(co2)
        fuel_history.append(fuel)

        print(f"Step {step} | Reward: {cumulative_reward:.1f} | Queue: {queue_len} | Wait: {avg_wait:.2f} s | Stops: {stops} | CO₂: {co2:.2f} | Fuel: {fuel:.2f}")

# Close SUMO
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


# Plotting
metrics = [
    (reward_history, "Cumulative Reward", "Cumulative Reward", 'blue'),
    (queue_history, "Total Queue Length", "Vehicles", 'orange'),
    (waiting_time_history, "Average Waiting Time", "Seconds", 'green'),
    (stops_history, "Number of Stops", "Stops", 'red'),
    (co2_history, "CO₂ Emissions", "mg", 'purple'),
    (fuel_history, "Fuel Consumption", "ml", 'brown'),
]

# Create figure
plt.figure(figsize=(12, 8))

# Plot each metric in a subplot
for i, (history, title, ylabel, color) in enumerate(metrics, 1):
    plt.subplot(3, 2, i)
    plt.plot(step_history, history, color=color)
    plt.xlabel("Simulation Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

# Add super title and layout
plt.suptitle("Actuated Traffic Light Metrics Over Simulation Steps", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save figure
plt.savefig(r"Athens Network\Case Kodrigktonos_Patision\Results\Actuated Traffic Light\Combined_Metrics.png", dpi=500, bbox_inches='tight')
plt.show()