"""
Reinforcement Fine-Tuning Trajectory Buffer Skill Client
Pure Python Standard Library implementation of Prioritized Trajectory Replay Buffer (Schaul et al.)
for Agent Reinforcement Fine-Tuning (RFT / GRPO).
Calculates group relative advantages, prioritized sampling probabilities, and importance weights.
"""

import math
from typing import List, Dict, Any, Tuple, Optional


class TrajectoryBuffer:
    """
    Prioritized Experience Replay Buffer storing full agent reasoning trajectories.
    """

    def __init__(self, capacity: int = 1000, alpha: float = 0.6, beta: float = 0.4):
        """
        :param capacity: Max trajectories stored.
        :param alpha: Priority exponent (0 = uniform, 1 = fully greedy).
        :param beta: Importance sampling exponent (0 = uncorrected, 1 = fully corrected).
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.trajectories: List[Dict[str, Any]] = []
        self.priorities: List[float] = []

    def add_trajectory(self, prompt: str, steps: List[Dict[str, Any]], final_reward: float, metadata: Optional[Dict[str, Any]] = None):
        """Add completed agent trajectory."""
        priority = (abs(final_reward) + 1e-5) ** self.alpha

        if len(self.trajectories) >= self.capacity:
            self.trajectories.pop(0)
            self.priorities.pop(0)

        traj = {
            "id": len(self.trajectories),
            "prompt": prompt,
            "steps": steps,
            "reward": final_reward,
            "metadata": metadata or {}
        }
        self.trajectories.append(traj)
        self.priorities.append(priority)

    def sample_batch(self, batch_size: int = 4) -> Dict[str, Any]:
        """
        Sample prioritized batch of trajectories with importance sampling weights.
        """
        n = len(self.trajectories)
        if n == 0:
            return {"trajectories": [], "weights": [], "indices": []}

        k = min(batch_size, n)
        total_p = sum(self.priorities)
        probs = [p / total_p for p in self.priorities]

        # Deterministic stratified or interval sampling
        step = total_p / k
        indices = []
        current_sum = 0.0
        cumsum = []
        for p in self.priorities:
            current_sum += p
            cumsum.append(current_sum)

        for i in range(k):
            target = (i + 0.5) * step
            idx = 0
            while idx < n - 1 and cumsum[idx] < target:
                idx += 1
            indices.append(idx)

        # Importance weights: w_i = (N * P(i)) ^ (- beta)
        max_w = 0.0
        weights = []
        for idx in indices:
            w = (n * probs[idx]) ** (-self.beta)
            weights.append(w)
            if w > max_w:
                max_w = w

        # Normalize weights
        norm_weights = [w / max_w for w in weights] if max_w > 0 else weights

        return {
            "trajectories": [self.trajectories[idx] for idx in indices],
            "weights": norm_weights,
            "indices": indices
        }

    def compute_group_relative_advantages(self, group_trajectories: List[Dict[str, Any]]) -> List[float]:
        """
        Compute GRPO (Group Relative Policy Optimization) normalized advantages:
        Advantage(i) = (Reward_i - Mean(Reward)) / (Std(Reward) + eps)
        """
        if not group_trajectories:
            return []

        rewards = [t["reward"] for t in group_trajectories]
        n = len(rewards)
        mean_r = sum(rewards) / n
        var = sum((r - mean_r) ** 2 for r in rewards) / n
        std_r = math.sqrt(var)

        advantages = []
        for r in rewards:
            adv = (r - mean_r) / (std_r + 1e-6)
            advantages.append(adv)

        return advantages
