from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from backend.Environment.SwarmEnv import SwarmEnv


class SimulationRequest(BaseModel):
    num_agents: int = Field(default=2, ge=1, le=50)
    space_size: float = Field(default=100.0, gt=0)
    max_episode_steps: int = Field(default=500, ge=1)
    seed: int | None = None
    wind_strength: float = Field(default=0.0, ge=0.0)
    wind_enabled: bool = True
    obstacle_count: int = Field(default=0, ge=0, le=100)
    dynamic_obstacles: bool = False
    policy_mode: str = Field(default="emergent")


class PolicyRequest(BaseModel):
    policy_mode: str = Field(default="emergent")


class CheckpointLoadRequest(BaseModel):
    checkpoint_path: str


class TrainingRequest(BaseModel):
    num_agents: int = Field(default=2, ge=1, le=50)
    iterations: int = Field(default=1, ge=1)
    checkpoint_dir: str = "checkpoints/api"


def compute_drone_actions(
    env: SwarmEnv,
    policy_mode: str = "emergent",
    mappo_agents: dict | None = None
) -> dict[str, np.ndarray]:
    actions = {}
    space_size = env.space_size

    if policy_mode == "random":
        for agent in env.agents:
            current_vel = env.drones[agent].velocity
            noise = np.random.uniform(-0.4, 0.4, size=3)
            actions[agent] = np.clip(current_vel + noise, -1.0, 1.0).astype(np.float32)
        return actions

    if policy_mode == "mappo" and mappo_agents:
        for agent in env.agents:
            if agent in mappo_agents:
                obs = env.get_observation(agent)
                actions[agent] = mappo_agents[agent].select_action(obs)
            else:
                actions[agent] = np.zeros(3, dtype=np.float32)
        return actions

    # Default / Emergent Multi-Agent Search Policy Controller
    for agent in env.agents:
        drone = env.drones[agent]
        pos = drone.get_position()
        vel = drone.velocity

        # 1. Neighbor Repulsion (Prevent collisions)
        repulsion = np.zeros(3, dtype=np.float32)
        for other_id, other_drone in env.drones.items():
            if other_id == agent:
                continue
            other_pos = other_drone.get_position()
            dist = np.linalg.norm(pos - other_pos)
            if dist < 12.0 and dist > 1e-4:
                dir_vec = (pos - other_pos) / dist
                weight = 1.0 / (dist ** 2 + 0.1)
                repulsion += dir_vec * weight * 18.0

        # 2. Dynamic Obstacle Avoidance
        obs_avoidance = np.zeros(3, dtype=np.float32)
        for obs in env.obstacles.values():
            obs_pos = obs.position
            dist = np.linalg.norm(pos - obs_pos)
            if dist < (obs.radius + 12.0) and dist > 1e-4:
                dir_vec = (pos - obs_pos) / dist
                weight = 1.0 / (dist ** 2 + 0.1)
                obs_avoidance += dir_vec * weight * 28.0

        # 3. Spatial Boundary Containment
        containment = np.zeros(3, dtype=np.float32)
        margin = 12.0
        for i in range(3):
            if pos[i] < margin:
                containment[i] += (margin - pos[i]) / margin
            elif pos[i] > space_size - margin:
                containment[i] -= (pos[i] - (space_size - margin)) / margin

        # 4. Spatial Exploration Force
        exploration = np.random.uniform(-0.3, 0.3, size=3).astype(np.float32)
        if np.linalg.norm(vel) > 0.05:
            exploration += (vel / np.linalg.norm(vel)) * 0.9
        else:
            # Spread drones outwards from center initial positions
            idx = int(agent.split("_")[-1]) if "_" in agent else 0
            angle = (idx / max(1, len(env.agents))) * 2.0 * np.pi
            exploration += np.array([np.cos(angle), np.sin(angle), np.sin(angle * 2.0) * 0.5], dtype=np.float32)

        # Total combined force vector
        total_force = (
            repulsion * 2.8 +
            obs_avoidance * 3.2 +
            containment * 2.2 +
            exploration * 1.2
        )

        target_action = np.clip(total_force, -1.0, 1.0)
        action = 0.65 * vel + 0.35 * target_action
        actions[agent] = np.clip(action, -1.0, 1.0).astype(np.float32)

    return actions


class SimulationManager:
    def __init__(self) -> None:
        self.environment: SwarmEnv | None = None
        self.status = "idle"
        self.policy_mode = "emergent"
        self.mappo_agents: dict = {}
        self.training_status = "idle"
        self.training_metrics: list[dict[str, Any]] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self.current_config: SimulationRequest = SimulationRequest()

    def reset(self, request: SimulationRequest) -> dict:
        self.stop()
        self.policy_mode = request.policy_mode
        self.current_config = request
        self.environment = SwarmEnv(
            num_agents=request.num_agents,
            space_size=request.space_size,
            max_episode_steps=request.max_episode_steps,
            wind_strength=request.wind_strength,
            wind_enabled=request.wind_enabled,
            obstacle_count=request.obstacle_count,
            dynamic_obstacles=request.dynamic_obstacles,
        )
        self.environment.reset(seed=request.seed)
        self.status = "running"
        return self.state()

    def step(self) -> dict:
        if self.environment is None:
            self.reset(self.current_config)
        assert self.environment is not None
        actions = compute_drone_actions(
            self.environment,
            policy_mode=self.policy_mode,
            mappo_agents=self.mappo_agents,
        )
        self.environment.step(actions)
        return self.state()

    def state(self) -> dict:
        if self.environment is None:
            return {
                "simulation_status": self.status,
                "episode": 0,
                "step": 0,
                "drones": [],
                "obstacles": [],
                "coverage": 0.0,
                "explored_cells": 0,
                "total_cells": 0,
                "collision_count": 0,
                "obstacle_collision_count": 0,
                "reward": 0.0,
                "total_reward": 0.0,
                "wind": [0.0, 0.0, 0.0],
                "curriculum_level": 1,
                "visited_cells": [],
                "policy_mode": self.policy_mode,
                "grid_size": 20,
            }
        return self.environment.get_simulation_state(policy_mode=self.policy_mode)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self.environment is not None and self.status != "idle":
            self.status = "stopped"

    def start(self, request: SimulationRequest = SimulationRequest()) -> dict:
        # If environment is not created, create it with request
        if self.environment is None:
            self.reset(request)
        # If request explicitly specifies non-default agent count different from active env, reset with request
        elif request.num_agents != 2 and request.num_agents != len(self.environment.agents):
            self.reset(request)

        self.status = "running"
        if self.environment is not None:
            self.environment.simulation_status = "running"

        self._stop_event.clear()
        if self._thread and self._thread.is_alive():
            return self.state()

        def run() -> None:
            while not self._stop_event.is_set() and self.environment is not None:
                state = self.step()
                if state["simulation_status"] == "completed":
                    break
                time.sleep(0.05)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return self.state()



manager = SimulationManager()
app = FastAPI(title="SwarmRL Backend API", version="1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/simulation/start")
def start_simulation(request: SimulationRequest = SimulationRequest()) -> dict:
    return manager.start(request)


@app.post("/simulation/stop")
def stop_simulation() -> dict:
    manager.stop()
    return manager.state()


@app.post("/simulation/reset")
def reset_simulation(request: SimulationRequest = SimulationRequest()) -> dict:
    return manager.reset(request)


@app.get("/simulation/state")
def simulation_state() -> dict:
    return manager.state()


@app.post("/simulation/policy")
def set_policy(request: PolicyRequest) -> dict:
    manager.policy_mode = request.policy_mode
    return manager.state()


@app.post("/simulation/load_checkpoint")
def load_checkpoint(request: CheckpointLoadRequest) -> dict:
    path = os.path.abspath(request.checkpoint_path)
    if not os.path.exists(path):
        return {"status": "error", "message": f"Checkpoint not found: {path}"}
    try:
        from backend.agents.mappo import MAPPOAgent
        if manager.environment is not None:
            for agent in manager.environment.agents:
                agent_obj = MAPPOAgent(agent_id=agent)
                # If agent weights file exists inside checkpoint directory
                agent_file = os.path.join(path, f"{agent}.pt")
                if os.path.isfile(agent_file):
                    agent_obj.load(agent_file)
                manager.mappo_agents[agent] = agent_obj
            manager.policy_mode = "mappo"
            return {"status": "success", "message": f"Loaded MAPPO checkpoint: {path}", "policy_mode": "mappo"}
    except Exception as err:
        return {"status": "error", "message": str(err)}
    return {"status": "success", "message": f"Checkpoint referenced: {path}", "policy_mode": "mappo"}


@app.post("/training/start")
def start_training(request: TrainingRequest) -> dict:
    if manager.training_status == "running":
        return {"status": "running"}

    def run_training() -> None:
        manager.training_status = "running"
        try:
            from backend.training.train import train
            checkpoint = train(
                num_agents=request.num_agents,
                iterations=request.iterations,
                checkpoint_dir=request.checkpoint_dir,
            )
            manager.training_metrics.append({"checkpoint": str(checkpoint)})
            manager.training_status = "completed"
        except Exception as error:
            manager.training_metrics.append({"error": str(error)})
            manager.training_status = "failed"

    threading.Thread(target=run_training, daemon=True).start()
    return {"status": "started"}


@app.post("/training/stop")
def stop_training() -> dict[str, str]:
    manager.training_status = "stopped"
    return {"status": manager.training_status}


@app.get("/training/status")
def training_status() -> dict[str, str]:
    return {"status": manager.training_status}


@app.get("/training/metrics")
def training_metrics() -> dict[str, list[dict[str, Any]]]:
    return {"metrics": manager.training_metrics}


@app.get("/checkpoints")
def checkpoints() -> dict[str, list[str]]:
    root = "checkpoints"
    paths = []
    if os.path.isdir(root):
        paths = [os.path.abspath(os.path.join(root, name)) for name in os.listdir(root)]
    return {"checkpoints": sorted(paths)}


@app.websocket("/ws/simulation")
async def simulation_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(manager.state())
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        return
    except Exception:
        await websocket.close()
