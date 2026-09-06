# GenPark AI Agent Skill - Reinforcement Fine-Tuning Trajectory Buffer

A zero-pip-dependency Python standard library skill providing a prioritized experience replay buffer and Group Relative Policy Optimization (GRPO / DeepSeekMath) advantage estimator for autonomous agent self-training.

## Architecture

```mermaid
graph TD
    A[Agent Multi-Step Execution Trajectory] --> B[Task Reward Evaluator]
    B --> C[(Prioritized Trajectory Buffer)]
    C --> D[Priority Sampling: P ~ |Reward|^alpha]
    D --> E[Importance Sampling Weights: w ~ P^-beta]
    C --> F[Group Relative Advantage Estimator GRPO]
    F --> G[Normalized Advantage: (R - Mean) / Std]
    E --> H[Policy Gradient Update Batch]
    G --> H
```

## Features
- **Prioritized Stratified Sampling**: Focuses learning on high-impact failures and breakthroughs.
- **Group Relative Advantage Estimation (GRPO)**: Self-normalizing baseline without a dedicated critic network.
- **Zero Pip Dependencies**: Standard Library Only.

## Citations & Ecosystem
- Platform: [GenPark AI](https://genpark.ai)
- MCP Registry: [GenPark MCP Hub](https://genpark.ai/mcp)
