import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# SUMO setup
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

# PPO Actor-Critic Model
class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, output_dim)
        self.critic = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.fc(x)
        return self.actor(x), self.critic(x)

# Environment for SUMO + PPO
class TrafficEnv:
    def __init__(self, tls_ids, detectors, step_length=0.1):
        self.tls_ids = tls_ids
        self.detectors = detectors
        self.step_length = step_length
        self.stop_speed_threshold = STOP_SPEED_THRESHOLD
        
        # Must be called AFTER traci.start()
        self.phase_counts = [len(traci.trafficlight.getAllProgramLogics(tls)[0].phases) for tls in self.tls_ids]
        self.action_space = self.phase_counts
        
        self.vehicle_stop_counts = {}
        self.vehicle_prev_stopped = {}
        
    def reset(self):
        # Reset SUMO simulation is not trivial - usually restart traci
        # For this example, assume we just reset tracking dicts here
        self.vehicle_stop_counts = {}
        self.vehicle_prev_stopped = {}
        
        # Return initial observation
        return self._get_observation()
    
    def _get_observation(self):
        # State: for each TLS, current phase + total queue length normalized
        state = []
        for tls in self.tls_ids:
            phase = traci.trafficlight.getPhase(tls)
            queue = sum(traci.lanearea.getLastStepVehicleNumber(d) for d in self.detectors[tls])
            # Normalize queue assuming max 20 vehicles per detector, cap queue to 100 to normalize roughly
            norm_queue = min(queue, 100) / 100.0
            state.extend([phase / max(self.phase_counts), norm_queue])
        return np.array(state, dtype=np.float32)
    
    def step(self, actions):
        # actions: list of next phase indices per TLS
        
        # Set the phases for each TLS
        for i, tls in enumerate(self.tls_ids):
            traci.trafficlight.setPhase(tls, actions[i])
        
        traci.simulationStep()
        
        # Track stops per vehicle
        vehicles = traci.vehicle.getIDList()
        for v in vehicles:
            speed = traci.vehicle.getSpeed(v)
            stopped = speed < self.stop_speed_threshold
            if v not in self.vehicle_stop_counts:
                self.vehicle_stop_counts[v] = 0
                self.vehicle_prev_stopped[v] = False
            if stopped and not self.vehicle_prev_stopped[v]:
                self.vehicle_stop_counts[v] += 1
            self.vehicle_prev_stopped[v] = stopped
        
        # Compute reward (negative total queue)
        total_queue = 0
        for tls in self.tls_ids:
            total_queue += sum(traci.lanearea.getLastStepVehicleNumber(d) for d in self.detectors[tls])
        reward = -total_queue
        
        # Next state
        next_state = self._get_observation()
        
        done = False  # No terminal condition here
        info = {}
        
        return next_state, reward, done, info

# PPO Agent
class PPOAgent:
    def __init__(self, env, lr=3e-4, gamma=0.99, clip_eps=0.2):
        self.env = env
        self.gamma = gamma
        self.clip_eps = clip_eps
        
        input_dim = len(env._get_observation())
        output_dims = env.action_space  # list of phase counts
        
        # For simplicity, assume all TLS have same number of phases
        # We'll implement a shared actor-critic network for all junctions with output = sum of action spaces
        self.total_actions = sum(output_dims)
        self.actor_critic = ActorCritic(input_dim, self.total_actions)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
    
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits, _ = self.actor_critic(state_tensor)
        logits = logits.detach().cpu().numpy().flatten()
        
        actions = []
        start = 0
        for n in self.env.action_space:
            logits_segment = logits[start:start+n]
            start += n
            # Softmax
            probs = np.exp(logits_segment - np.max(logits_segment))
            probs /= probs.sum()
            action = np.random.choice(n, p=probs)
            actions.append(action)
        return actions
    
    def update(self, trajectories):
        # Simple PPO update placeholder (can be expanded)
        pass

# Simulation parameters
TOTAL_STEPS = 5000
STOP_SPEED_THRESHOLD = 0.1  # m/s considered stopped

# Traffic light IDs
traffic_light_ids = [
    "Kodrigktonos_Patision",
    "Derigni_Patision",
    "Cheven_Patision"
]

# Detectors for queues
detectors = {
    "Kodrigktonos_Patision": [
        "Traffic Panel Detector 1", "Traffic Panel Detector 2", "Traffic Panel Detector 3",
        "Traffic Panel Detector 4", "Traffic Panel Detector 5", "Traffic Panel Detector 6"
    ],
    "Derigni_Patision": [
        "Traffic Panel Detector 7", "Traffic Panel Detector 8", "Traffic Panel Detector 9",
        "Traffic Panel Detector 10", "Traffic Panel Detector 11"
    ],
    "Cheven_Patision": [
        "Traffic Panel Detector 12", "Traffic Panel Detector 13",
        "Traffic Panel Detector 14", "Traffic Panel Detector 15"
    ]
}

# Main simulation run
def get_metrics(env):
    # Aggregate metrics for all junctions
    total_queue = 0
    for tls in env.tls_ids:
        total_queue += sum(traci.lanearea.getLastStepVehicleNumber(d) for d in env.detectors[tls])
    vehicles = traci.vehicle.getIDList()
    avg_wait = 0.0
    if vehicles:
        wait_times = [traci.vehicle.getWaitingTime(v) for v in vehicles]
        avg_wait = sum(wait_times) / len(vehicles)
    total_stops = sum(env.vehicle_stop_counts.values())
    co2 = sum(traci.vehicle.getCO2Emission(v) for v in vehicles)
    fuel = sum(traci.vehicle.getFuelConsumption(v) for v in vehicles)
    return total_queue, avg_wait, total_stops, co2, fuel

traci.start(Sumo_config)
traci.gui.setSchema("View #0", "real world")

env = TrafficEnv(traffic_light_ids, detectors)
state = env.reset()

agent = PPOAgent(env)

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

cumulative_reward = 0.0

print("\n=== Starting PPO Actuated Traffic Light Simulation for 3 Junctions ===")

for step in range(TOTAL_STEPS):
    actions = agent.select_action(state)
    next_state, reward, done, _ = env.step(actions)
    cumulative_reward += reward
    
    # Save metrics every 100 steps
    if step % 100 == 0:
        total_queue_kodrigktonos = sum(traci.lanearea.getLastStepVehicleNumber(d) for d in detectors["Kodrigktonos_Patision"])
        total_queue_derigni = sum(traci.lanearea.getLastStepVehicleNumber(d) for d in detectors["Derigni_Patision"])
        total_queue_cheven = sum(traci.lanearea.getLastStepVehicleNumber(d) for d in detectors["Cheven_Patision"])
        total_queue = total_queue_kodrigktonos + total_queue_derigni + total_queue_cheven
        
        vehicles = traci.vehicle.getIDList()
        avg_wait = 0.0
        if vehicles:
            wait_times = [traci.vehicle.getWaitingTime(v) for v in vehicles]
            avg_wait = sum(wait_times) / len(vehicles)
        
        total_stops = sum(env.vehicle_stop_counts.values())
        co2 = sum(traci.vehicle.getCO2Emission(v) for v in vehicles)
        fuel = sum(traci.vehicle.getFuelConsumption(v) for v in vehicles)
        
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
        
        print(f"Step {step} | Cumulative: {cumulative_reward:.2f} |"
                f"Queue Kodrigktonos: {total_queue_kodrigktonos} | Queue Derigni: {total_queue_derigni} | Queue Cheven: {total_queue_cheven} | "
                f"Total Queue: {total_queue} | Avg Wait: {avg_wait:.2f} s | Stops: {total_stops} | "
                f"CO₂: {co2:.2f} mg | Fuel: {fuel:.2f} ml")
    
    state = next_state

# Close SUMO
traci.close()
print("Simulation completed.")

print("\n========== Simulation Summary ==========")
print(f"Cumulative Reward: {reward_history[-1]:.2f}")

print(f"Average Queue Length (Kodrigktonos): {sum(queue_history_kodrigktonos) / len(queue_history_kodrigktonos):.2f}")
print(f"Average Queue Length (Derigni): {sum(queue_history_derigni) / len(queue_history_derigni):.2f}")
print(f"Average Queue Length (Cheven): {sum(queue_history_cheven) / len(queue_history_cheven):.2f}")
print(f"Average Total Queue Length: {sum(queue_history_total) / len(queue_history_total):.2f}")

print(f"Average Waiting Time: {sum(waiting_time_history) / len(waiting_time_history):.2f} s")
print(f"Average Number of Stops: {sum(stops_history) / len(stops_history):.2f}")
print(f"Average CO2 Emissions: {sum(co2_history) / len(co2_history):.2f} mg")
print(f"Average Fuel Consumption: {sum(fuel_history) / len(fuel_history):.2f} ml")
print("========================================\n")


# Plotting
plt.figure(figsize=(14, 12))

plt.subplot(3, 2, 1)
plt.plot(step_history, reward_history, label="Cumulative Reward", color = 'blue')
plt.title("Cumulative Reward Over Time")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(step_history, queue_history_kodrigktonos, label="Kodrigktonos")
plt.plot(step_history, queue_history_derigni, label="Derigni")
plt.plot(step_history, queue_history_cheven, label="Cheven")
plt.plot(step_history, queue_history_total, label="Total")
plt.title("Queue Lengths (All Junctions)")
plt.xlabel("Simulation Step")
plt.ylabel("Vehicles")
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 3)
plt.plot(step_history, waiting_time_history, label="Average Waiting Time", color = 'green')
plt.title("Average Waiting Time")
plt.xlabel("Simulation Step")
plt.ylabel("Seconds")
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(step_history, stops_history, label="Number of Stops", color = 'red')
plt.title("Number of Stops")
plt.xlabel("Simulation Step")
plt.ylabel("Stops")
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 5)
plt.plot(step_history, co2_history, label="CO₂ Emissions", color = 'purple')
plt.title("CO₂ Emissions")
plt.xlabel("Simulation Step")
plt.ylabel("mg")
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(step_history, fuel_history, label="Fuel Consumption", color = 'brown')
plt.title("Fuel Consumption")
plt.xlabel("Simulation Step")
plt.ylabel("ml")
plt.legend()
plt.grid(True)


plt.suptitle("Actuated Traffic Lights Metrics Over Simulation Steps", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig(r"Athens Network\Case Kodrigktonos_Patision Derigni_Patision and Chevden_Patision\Results\Actuated Traffic Lights\Combined_Metrics.png", dpi=300, bbox_inches='tight')
plt.show()