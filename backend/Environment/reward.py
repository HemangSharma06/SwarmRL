class SwarmReward:

    def __init__(
        self,
        collision_penalty=-100.0,
        proximity_penalty=-1.0,
        safe_reward=1.0,
        proximity_threshold=10.0
    ):
        self.collision_penalty = collision_penalty
        self.proximity_penalty = proximity_penalty
        self.safe_reward = safe_reward
        self.proximity_threshold = proximity_threshold

    def calculate_reward(
        self,
        collision,
        distance_to_nearest
    ):

        # Collision
        if collision:
            return self.collision_penalty

        # Too close to another drone
        if distance_to_nearest < self.proximity_threshold:
            return self.proximity_penalty

        # Normal safe movement
        return self.safe_reward