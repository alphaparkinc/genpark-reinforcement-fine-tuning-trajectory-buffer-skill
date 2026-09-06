"""
Example usage of Reinforcement Fine-Tuning Trajectory Buffer Skill.
"""

from client import TrajectoryBuffer


def main():
    print("=== Reinforcement Fine-Tuning Trajectory Buffer Demonstration ===")
    buffer = TrajectoryBuffer(capacity=100)

    # Insert simulated agent trajectories with varying task rewards
    buffer.add_trajectory(
        prompt="Write a fast prime sieve",
        steps=[{"action": "tool_exec", "code": "def sieve(n): ..."}],
        final_reward=1.0
    )
    buffer.add_trajectory(
        prompt="Write a fast prime sieve",
        steps=[{"action": "tool_exec", "code": "slow trial division"}],
        final_reward=-0.5
    )
    buffer.add_trajectory(
        prompt="Refactor database schema",
        steps=[{"action": "sql_query", "sql": "CREATE INDEX ..."}],
        final_reward=0.8
    )
    buffer.add_trajectory(
        prompt="Deploy serverless function",
        steps=[{"action": "cloud_api", "status": "timeout"}],
        final_reward=-1.0
    )

    # Sample prioritized batch
    batch = buffer.sample_batch(batch_size=2)
    print(f"Sampled {len(batch['trajectories'])} Prioritized Trajectories:")
    for t, w in zip(batch["trajectories"], batch["weights"]):
        print(f"  Prompt: '{t['prompt']}' | Reward: {t['reward']} | Importance Weight: {w:.4f}")

    # Compute Group Relative Advantages (GRPO)
    print("\n--- Group Relative Policy Optimization (GRPO) Advantages ---")
    group = [
        {"reward": 1.0},
        {"reward": 0.5},
        {"reward": -0.5},
        {"reward": -1.0}
    ]
    advantages = buffer.compute_group_relative_advantages(group)
    for idx, (t, adv) in enumerate(zip(group, advantages)):
        print(f"  Candidate #{idx + 1}: Raw Reward = {t['reward']:+.1f} -> Normalized Advantage = {adv:+.4f}")


if __name__ == "__main__":
    main()
