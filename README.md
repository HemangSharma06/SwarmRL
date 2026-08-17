# SwarmRL — Multi-Agent Deep Reinforcement Learning Simulator

## Overview

**SwarmRL** is a Multi-Agent Deep Reinforcement Learning system designed to train a swarm of autonomous drones to collaboratively search and cover a disaster-affected environment.

Instead of manually defining paths for each drone, SwarmRL allows the drones to **learn their behavior through reinforcement learning**. Each drone observes its surroundings, takes an action, and receives rewards or penalties based on its behavior.

The main goal is to make the swarm efficiently explore the environment while **avoiding collisions and maximizing collective coverage**.

---
## Objectives

The primary objectives of SwarmRL are:

- Train multiple autonomous drones simultaneously.
- Maximize the overall area explored by the swarm.
- Minimize collisions between drones.
- Keep drones within the environment boundaries.
- Encourage coordinated and efficient swarm behavior.
- Learn the optimal behavior instead of manually programming drone paths.
- Enable the swarm to adapt to dynamic obstacles and environmental conditions.
---

## Concept
SwarmRL is based on **Multi-Agent Reinforcement Learning (MARL)**.
Each drone is treated as an independent agent that interacts with the environment.
The basic learning cycle is:
```text
Observe → Decide → Act → Receive Reward → Learn → Repeat
