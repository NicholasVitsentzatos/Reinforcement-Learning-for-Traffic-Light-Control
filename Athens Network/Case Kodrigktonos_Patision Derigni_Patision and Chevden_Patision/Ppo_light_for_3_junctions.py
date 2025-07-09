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
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

# SUMO Config
sumo_cfg = [
    'sumo-gui', 
    '-c', 'Athens Network\Data\osm.sumocfg', 
    '--step-length', '0.10', 
    '--delay', '50', 
    '--lateral-resolution', '0'
]
traci.start(sumo_cfg)
traci.gui.setSchema("View #0", "real world")

# PPO Constants
TOTAL_STEPS = 5000
GAMMA = 0.99
EPS_CLIP = 0.2
LR = 3e-4
UPDATE_INTERVAL = 200
MIN_GREEN = 100
STOP_THRESH = 0.1

# Junction Definitions
TLS_KP, TLS_DP, TLS_CP = "Kodrigktonos_Patision", "Derigni_Patision", "Cheven_Patision"
KP_dets = [f"Traffic Panel Detector {i}" for i in range(1,7)]
DP_dets = [f"Traffic Panel Detector {i}" for i in range(7,12)]
CP_dets = [f"Traffic Panel Detector {i}" for i in range(12,16)]

# PPO Model
class PolicyNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU())
        self.act = nn.Linear(64,2)
        self.val = nn.Linear(64,1)
    def forward(self, x):
        x = self.fc(x)
        return self.act(x), self.val(x)

# PPO Agent
class PPOAgent:
    def __init__(self, obs_dim):
        self.policy = PolicyNet(obs_dim)
        self.old = PolicyNet(obs_dim)
        self.old.load_state_dict(self.policy.state_dict())
        self.opt = optim.Adam(self.policy.parameters(), lr=LR)
        self.MSE = nn.MSELoss()

    def select(self, state):
        s = torch.FloatTensor(state)
        logits, _ = self.old(s)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return a.item(), dist.log_prob(a)

    def update(self, mem):
        S, A, R, S2, LP, D = zip(*mem)
        S = torch.FloatTensor(S)
        A = torch.LongTensor(A)
        LP = torch.stack(LP).detach()
        Rets = []
        G = 0
        for r, d in zip(reversed(R), reversed(D)):
            G = r + GAMMA * G * (1 - d)
            Rets.insert(0, G)
        Rets = torch.FloatTensor(Rets)

        for _ in range(4):
            logits, V = self.policy(S)
            dist = Categorical(logits=logits)
            LP_new = dist.log_prob(A)
            ent = dist.entropy().mean()
            ratio = (LP_new - LP).exp()
            adv = Rets - V.squeeze().detach()
            loss = -torch.min(ratio * adv, torch.clamp(ratio, 1-EPS_CLIP,1+EPS_CLIP) * adv).mean() \
                   + 0.5*self.MSE(V.squeeze(), Rets) - 0.01*ent

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        self.old.load_state_dict(self.policy.state_dict())

# Helper Functions
def get_queue(dets):
    return [traci.lanearea.getLastStepVehicleNumber(d) for d in dets]

def get_state(dets, tls):
    q = get_queue(dets)
    p = traci.trafficlight.getPhase(tls)
    return q + [p]

def get_reward(s):
    return -sum(s[:-1])

def apply_action(a, tls, step, last):
    if a == 1 and step - last >= MIN_GREEN:
        phs = traci.trafficlight.getAllProgramLogics(tls)[0].phases
        next_ph = (traci.trafficlight.getPhase(tls)+1) % len(phs)
        traci.trafficlight.setPhase(tls, next_ph)
        return step
    return last

def get_metrics(dets):
    w, st, c, f = 0.0, 0, 0.0, 0.0
    seen = set()
    for d in dets:
        for vid in traci.lanearea.getLastStepVehicleIDs(d):
            if vid not in seen:
                seen.add(vid)
                w += traci.vehicle.getWaitingTime(vid)
                sp = traci.vehicle.getSpeed(vid)
                c += traci.vehicle.getCO2Emission(vid)
                f += traci.vehicle.getFuelConsumption(vid)
                if sp < STOP_THRESH: st += 1
    return w, st, c, f

# Agents & Memory
A_kp, A_dp, A_cp = PPOAgent(7), PPOAgent(6), PPOAgent(5)
M_kp, M_dp, M_cp = [], [], []
L_kp, L_dp, L_cp = -MIN_GREEN, -MIN_GREEN, -MIN_GREEN
cum_reward = 0.0

# Metrics records
steps, rewards, queues = [], [], []
wk, wd, wc = [], [], []
sk, sd, sc = [], [], []
ek, ed, ec = [], [], []
fk, fd, fc = [], [], []

# Main Loop
print("\n=== PPO Training for 3 Junctions ===")
for t in range(TOTAL_STEPS):
    s1 = get_state(KP_dets, TLS_KP)
    s2 = get_state(DP_dets, TLS_DP)
    s3 = get_state(CP_dets, TLS_CP)

    a1, lp1 = A_kp.select(s1)
    a2, lp2 = A_dp.select(s2)
    a3, lp3 = A_cp.select(s3)

    L_kp = apply_action(a1, TLS_KP, t, L_kp)
    L_dp = apply_action(a2, TLS_DP, t, L_dp)
    L_cp = apply_action(a3, TLS_CP, t, L_cp)

    traci.simulationStep()

    s1n = get_state(KP_dets, TLS_KP)
    s2n = get_state(DP_dets, TLS_DP)
    s3n = get_state(CP_dets, TLS_CP)

    r1, r2, r3 = get_reward(s1n), get_reward(s2n), get_reward(s3n)
    cum_reward += (r1 + r2 + r3)

    M_kp.append((s1, a1, r1, s1n, lp1, False))
    M_dp.append((s2, a2, r2, s2n, lp2, False))
    M_cp.append((s3, a3, r3, s3n, lp3, False))

    if (t+1) % UPDATE_INTERVAL == 0:
        A_kp.update(M_kp); M_kp.clear()
        A_dp.update(M_dp); M_dp.clear()
        A_cp.update(M_cp); M_cp.clear()

    w1, s1s, c1, f1 = get_metrics(KP_dets)
    w2, s2s, c2, f2 = get_metrics(DP_dets)
    w3, s3s, c3, f3 = get_metrics(CP_dets)

    if t % 100 == 0:
        steps.append(t)
        rewards.append(cum_reward)
        queues.append(sum(s1n[:-1]) + sum(s2n[:-1]) + sum(s3n[:-1]))

        wk.append(w1); wd.append(w2); wc.append(w3)
        sk.append(s1s); sd.append(s2s); sc.append(s3s)
        ek.append(c1); ed.append(c2); ec.append(c3)
        fk.append(f1); fd.append(f2); fc.append(f3)

        print(
            f"\nStep {t} Metrics:"
            f"\n  Cumulative Reward: {cum_reward:.2f}  |  Total Queue: {queues[-1]}"
            f"\n  [Kodrigktonos_Patision]  Reward: {r1:.2f}  |  Waiting: {w1:.2f}s  |  Stops: {s1s}  |  CO₂: {c1:.2f}g  |  Fuel: {f1:.2f}L"
            f"\n  [Derigni_Patision]       Reward: {r2:.2f}  |  Waiting: {w2:.2f}s  |  Stops: {s2s}  |  CO₂: {c2:.2f}g  |  Fuel: {f2:.2f}L"
            f"\n  [Cheven_Patision]        Reward: {r3:.2f}  |  Waiting: {w3:.2f}s  |  Stops: {s3s}  |  CO₂: {c3:.2f}g  |  Fuel: {f3:.2f}L"
        )

traci.close()

print("\n========== Simulation Summary ==========")
print(f"Cumulative Reward: {rewards[-1]:.2f}")

print(f"Average Total Queue Length: {sum(queues) / len(queues):.2f}")

print(f"Average Waiting Time (Kodrigktonos): {sum(wk) / len(wk):.2f} s")
print(f"Average Waiting Time (Derigni): {sum(wd) / len(wd):.2f} s")
print(f"Average Waiting Time (Cheven): {sum(wc) / len(wc):.2f} s")

print(f"Average Stops (Kodrigktonos): {sum(sk) / len(sk):.2f}")
print(f"Average Stops (Derigni): {sum(sd) / len(sd):.2f}")
print(f"Average Stops (Cheven): {sum(sc) / len(sc):.2f}")

print(f"Average CO2 Emissions (Kodrigktonos): {sum(ek) / len(ek):.2f} mg")
print(f"Average CO2 Emissions (Derigni): {sum(ed) / len(ed):.2f} mg")
print(f"Average CO2 Emissions (Cheven): {sum(ec) / len(ec):.2f} mg")

print(f"Average Fuel Consumption (Kodrigktonos): {sum(fk) / len(fk):.2f} ml")
print(f"Average Fuel Consumption (Derigni): {sum(fd) / len(fd):.2f} ml")
print(f"Average Fuel Consumption (Cheven): {sum(fc) / len(fc):.2f} ml")
print("==============================================\n")


# Plotting
plt.figure(figsize=(14, 12))

plt.subplot(3, 2, 1)
plt.plot(steps, rewards, label="Total Reward")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Cumulative Reward")
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(steps, queues, label="Total Queue Length")
plt.xlabel("Step")
plt.ylabel("Vehicles")
plt.title("Total Queue")
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 3)
plt.plot(steps, wk, label="Kodrigktonos (KP)")
plt.plot(steps, wd, label="Derigni (DP)")
plt.plot(steps, wc, label="Cheven (CP)")
plt.xlabel("Step")
plt.ylabel("Seconds")
plt.title("Average Waiting Time")
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(steps, sk, label="Kodrigktonos (KP)")
plt.plot(steps, sd, label="Derigni (DP)")
plt.plot(steps, sc, label="Cheven (CP)")
plt.xlabel("Step")
plt.ylabel("Count")
plt.title("Number of Stops")
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 5)
plt.plot(steps, ek, label="Kodrigktonos (KP)")
plt.plot(steps, ed, label="Derigni (DP)")
plt.plot(steps, ec, label="Cheven (CP)")
plt.xlabel("Step")
plt.ylabel("CO₂ (mg)")
plt.title("CO₂ Emissions")
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 6)
plt.plot(steps, fk, label="Kodrigktonos (KP)")
plt.plot(steps, fd, label="Derigni (DP)")
plt.plot(steps, fc, label="Cheven (CP)")
plt.xlabel("Step")
plt.ylabel("Fuel (ml)")
plt.title("Fuel Consumption")
plt.grid(True)
plt.legend()

plt.suptitle("PPO Traffic Lights Metrics Over Simulation Steps", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig(r"Athens Network\Case Kodrigktonos_Patision Derigni_Patision and Chevden_Patision\Results\PPO Traffic Lights\Combined_Metrics.png", dpi=300, bbox_inches='tight')
plt.show()