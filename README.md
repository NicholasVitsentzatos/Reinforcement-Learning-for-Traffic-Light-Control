# 🚦 Intelligent Traffic Signal Control Using SUMO and Reinforcement Learning

This repository explores the design, implementation, and evaluation of intelligent traffic signal control methods using the **SUMO** (Simulation of Urban MObility) traffic simulator. It focuses on optimizing traffic flow in both **real-world** and **synthetic** road networks using a range of traditional and AI-based control strategies.

---

## 📌 Project Overview

The project investigates several traffic control techniques, including:

- **Fixed-Time Control (Typical Light)**: Predefined light schedules.
- **Actuated Control**: Reactive signal changes based on detector inputs.
- **Reinforcement Learning Approaches**:
  - Single-agent RL
  - Multi-agent RL (with and without communication)
  - Policy optimization techniques (e.g., PPO)

All methods are tested under consistent conditions and evaluated using simulated metrics.

---

## 🗺️ Target Networks

Simulations are carried out in **two types of networks**:

### 🏙 Real Network (Athens)
A realistic urban layout based on traffic intersections in **Athens, Greece**, featuring:

- **Kodrigktonos–Patision**
- **Derigni–Patision**
- **Cheven**

### 🧪 Synthetic Network
A smaller, controlled testbed designed for **rapid prototyping** and **experimentation**, allowing faster iteration and testing of new ideas in isolation before deploying to real scenarios.

---

## 📊 Evaluation Metrics

Performance is assessed with a comprehensive set of metrics:

- **Cumulative Reward** (when applicable)
- **Average Queue Length**
- **Average Waiting Time**
- **Average Number of Stops**
- **CO₂ Emissions**
- **Fuel Consumption**

> Metrics are collected either **per junction** or **combined**, depending on the method and scope.

---

## 📁 Repository Structure

├── Synthetic Network/ # Experimental testbed

│    ├── Data/ # Synthetic SUMO network files (Must be downloaded from my shared google drive folder)

│    ├── Results

│    ├── Typical_light_controller.py

│    ├── Q_light_controller.py

│    ├── Deep_Q_light_controller.py

│    └── Synthetick Network Picture
    
├── Athens Network
│    ├── Case Kondriktonos_Patision
│       ├── Results
│       ├── Typical_light_controller.py
│        ├── Q_light_controller.py
│        ├── Deep_Q_light_controller.py
│        ├── PPO_light_controller.py
│        └── Actuated_light_controller.py
│    ├── Case Kondriktonos_Derigni_Patision
│        ├── Results
│        ├── Typical_light_for_2_junctions.py
│        ├── Q_light_for_2_junctions.py
│        ├── PPO_light_for_2_junctions.py
│        ├── Actuated_light_controller.py
│        ├── Multi_Agents__for_2_junctions.py
│        └── Multi_Agents__with_communication_for_2_junctions.py
│    ├── Case Kondriktonos_Derigni_Cheven_Patision
│        ├── Results
│        ├── Typical_light_for_3_junctions.py
│        ├── Q_light__for_3_junctions.py
│        ├── PPO_light_for_3_junctions.py
│        ├── Actuated_light_controller.py
│        ├── Multi_Agents__for_3_junctions.py
│        └── Multi_Agents__with_communication_for_3_junctions.py
│    ├── Referenced City Network Picture
│    └── Simulated City Network Picture
│├── Data/ # Athens SUMO network files (Must be downloaded from my shared google drive folder)
│└── README.md

---

## ⚙️ Getting Started

1. Install [SUMO](https://sumo.dlr.de/docs/Downloads.html) and set the `SUMO_HOME` environment variable.
2. Install required Python packages:
   ```bash
   pip install traci
