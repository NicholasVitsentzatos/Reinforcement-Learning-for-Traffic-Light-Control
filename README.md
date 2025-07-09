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

    ├── Data/ # Synthetic SUMO network files (Must be downloaded from my shared google drive folder)

    ├── Results
    
    ├── Typical_light_controller.py
    
    ├── Q_light_controller.py
    
    ├── Deep_Q_light_controller.py
    
    └── Synthetick Network Picture
    
├── Athens Network

    ├── Case Kondriktonos_Patision

       ├── Results

       ├── Typical_light_controller.py

        ├── Q_light_controller.py

        ├── Deep_Q_light_controller.py

        ├── PPO_light_controller.py

        └── Actuated_light_controller.py

    ├── Case Kondriktonos_Derigni_Patision

        ├── Results

        ├── Typical_light_for_2_junctions.py

        ├── Q_light_for_2_junctions.py

        ├── PPO_light_for_2_junctions.py

        ├── Actuated_light_controller.py

        ├── Multi_Agents__for_2_junctions.py

        └── Multi_Agents__with_communication_for_2_junctions.py

    ├── Case Kondriktonos_Derigni_Cheven_Patision

        ├── Results

        ├── Typical_light_for_3_junctions.py

        ├── Q_light__for_3_junctions.py

        ├── PPO_light_for_3_junctions.py

        ├── Actuated_light_controller.py

        ├── Multi_Agents__for_3_junctions.py

        └── Multi_Agents__with_communication_for_3_junctions.py

    ├── Referenced City Network Picture

    └── Simulated City Network Picture

├── Data/ # Athens SUMO network files (Must be downloaded from my shared google drive folder)

└── README.md

---

## ⚙️ Getting Started

1. Install [SUMO](https://sumo.dlr.de/docs/Downloads.html) and set the `SUMO_HOME` environment variable.
2. Install required Python packages:
   ```bash
   pip install traci torch

## Results

### Synthetic Network 

| Method         | Cumulative Reward | Average Queue Length  |
|----------------|-------------------|-----------------------|
| Typical Light  | -24,209.00        | 4.90                  |
| Q Light        | -17,642.00        | 3.48                  |
| Deep Q Light   | -16,358.00        | 3.32                  |


### Kondriktonos - Patision 

| Method         | Cumulative Reward | Avg Queue Length | Avg Waiting Time (s) | Avg Stops | Avg CO₂ Emissions (mg) | Avg Fuel Consumption (ml)    |
|----------------|-------------------|------------------|-----------------------|-----------|-------------------------|----------------------------|
| Typical Light  | -33,042.00        | 6.66             | 29.05                 | 1,364.86  | 251,864.93              | 81,447.07                  |
| Q Light        | -27,860.00        | 5.80             | 18.95                 | 1,455.16  | 239,822.02              | 77,534.58                  |
| Actuated Light | -30,800.00        | 6.26             | 20.23                 | 1,526.02  | 240,167.18              | 77,656.44                  |
| PPO Light      | -29,640.00        | 6.46             | 20.43                 | 1,455.54  | 237,014.06              | 76,623.15                  |
| Deep Q Light   | -29,129.00        | 5.86             | 22.99                 |   892.04  | 246,175.09              | 79,587.04                  |


### Kondriktonos - Derigni - Patision 


| Method                         | Cumulative Reward                          | Avg Queue Length                                 | Avg Waiting Time (s)                                  | Avg Stops                                     | Avg CO₂ Emissions (mg)                                  | Avg Fuel Consumption (ml)                               |
|--------------------------------|--------------------------------------------|--------------------------------------------------|-------------------------------------------------------|-----------------------------------------------|---------------------------------------------------------|---------------------------------------------------------|
| Typical Light                  | -55,768.00                                 | K: 6.06 + D: 5.34 = 11.40                        | 9.93                                                  | 2,125.14                                      | 393,671.36                                              | 127,418.74                                              |
| Q Light                        | -65,456.00                                 | 13.68                                            | K: 210.66 + D: 24.42 = 235.08                         | K: 3.94 + D: 4.28 = 8.22                      | K: 9,670.79 + D: 14,149.96 = 23,820.75                  | K: 3,127.29 + D: 4,584.87 = 7,712.16                    |
| Actuated Light                 | -70,174.00                                 | K: 6.20 + D: 8.04 = 14.24                        | 20.90                                                 | 2,132.96                                      | 424,309.33                                              | 137,341.99                                              |
| PPO Light                      | -67,304.00                                 | 14.32                                            | K: 172.87 + D: 22.65 = 195.52                         | K: 4.20 + D: 4.56 = 8.76                      | K: 13,107.11 + D: 14,958.62 = 28,065.73                 | K: 4,244.83 + D: 4,845.18 = 9,090.01                    |
| Multi Agent                    | K: -27393.00 + D: -34401.00 = -61794.0     | K: 5.58 + D: 6.92 = 12.50                        | K: 163.86 + D: 18.87 = 182.73                         | K: 4.60 + D: 3.70 = 8.30                      | K: 9,806.12 + D: 12,745.34 = 22,551.46                  | K: 3,173.03 + D: 4,128.98 = 7,302.01                    |
| Multi Agent w/ Communication   | K: -28790.00 + D: -35674.00 = -64464.0     | K: 4.28 + D: 6.72 = 11.00                        | K: 114.94 + D: 14.70 = 129.64                         | K: 2.92 + D: 3.44 = 6.36                      | K: 9,062.22 + D: 11,053.29 = 20,115.51                  | K: 2,929.46 + D: 3,581.93 = 6,511.39                    |


### Kondriktonos - Derigni - Cheven - Patision 


| Method                         | Cumulative Reward                                   | Avg Queue Length                                           | Avg Waiting Time (s)                                                   | Avg Stops                                             | Avg CO₂ Emissions (mg)                                             | Avg Fuel Consumption (ml)                                         |
|--------------------------------|-----------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------------------|-------------------------------------------------------|--------------------------------------------------------------------|-------------------------------------------------------------------|
| Typical Light                  | -79467.00                                           | K: 6.06 + D: 5.34 + C: 4.78 = 16.18                        | 9.93                                                                   | 2,125.14                                              | 393,671.36                                                         | 127,418.74                                                        |
| Q Light                        | -98184.00                                           | 19.68                                                      | K: 150.61 + D: 21.34 + C: 5.55 = 177.50                                | K: 3.78 + D: 4.26 + C: 1.86 = 9.90                    | K: 8,899.43 + D: 13,407.49 + C: 9,665.57 = 31,972.49               | K: 2,879.23 + D: 4,344.30 + C: 3,133.14 = 10,356.67               |
| Actuated Light                 | -96558.00                                           | K: 6.14 + D: 7.64 + C: 4.92 = 18.70                        | 9.89                                                                   | 3,634.02                                              | 417,260.95                                                         | 135,085.96                                                        |
| PPO Light                      | -89179.00                                           | 17.98                                                      | K: 114.80 + D: 21.29 + C: 6.86 = 142.95                                | K: 4.06 + D: 4.08 + C: 2.20 = 10.34                   | K: 10,073.97 + D: 14,315.26 + C: 8,909.72 = 33,298.95              | K: 3,261.41 + D: 4,638.40 + C: 2,887.51 = 10,787.32               |
| Multi Agent                    | K: -27771.00 + D: -37861.00 + C: -24293.00 = -89925.0  | K: 6.24 + D: 7.74 + C: 4.38 = 18.36                        | K: 244.69 + D: 17.51 + C: 3.64 = 265.84                                | K: 5.12 + D: 4.62 + C: 1.42 = 11.16                   | K: 10,463.25 + D: 15,444.17 + C: 8,784.38 = 34,691.80              | K: 3,384.16 + D: 5,002.80 + C: 2,846.77 = 11,233.73               |
| Multi Agent w/ Communication   | K: -31301.00 + D: -37460.00 + C: -22692.00 = -91453.0  | K: 5.82 + D: 6.88 + C: 4.66 = 17.36                        | K: 221.08 + D: 20.36 + C: 5.14 = 246.58                                | K: 3.98 + D: 4.58 + C: 1.90 = 10.46                   | K: 10,608.19 + D: 15,874.19 + C: 9,979.13 = 36,461.51              | K: 3,434.35 + D: 5,143.33 + C: 3,234.09 = 11,811.77               |
