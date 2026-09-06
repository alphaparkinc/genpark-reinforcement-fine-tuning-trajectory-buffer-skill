"""
MCP Server for Reinforcement Fine-Tuning Trajectory Buffer Skill.
"""

import json
import sys
from client import TrajectoryBuffer

BUFFER = TrajectoryBuffer()


def handle_request(req: dict) -> dict:
    method = req.get("method")
    params = req.get("params", {})

    if method == "tools/list":
        return {
            "tools": [
                {
                    "name": "add_trajectory",
                    "description": "Store an agent trajectory with task reward",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "steps": {"type": "array"},
                            "final_reward": {"type": "number"}
                        },
                        "required": ["prompt", "steps", "final_reward"]
                    }
                },
                {
                    "name": "sample_batch",
                    "description": "Sample prioritized trajectories with importance sampling weights",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "batch_size": {"type": "integer", "default": 4}
                        }
                    }
                },
                {
                    "name": "compute_grpo_advantages",
                    "description": "Calculate group relative policy optimization normalized advantages",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "group_trajectories": {"type": "array", "items": {"type": "object"}}
                        },
                        "required": ["group_trajectories"]
                    }
                }
            ]
        }
    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})

        if tool_name == "add_trajectory":
            BUFFER.add_trajectory(
                args["prompt"],
                args["steps"],
                args["final_reward"]
            )
            return {"content": [{"type": "text", "text": json.dumps({"status": "trajectory_added"})}]}

        elif tool_name == "sample_batch":
            res = BUFFER.sample_batch(args.get("batch_size", 4))
            return {"content": [{"type": "text", "text": json.dumps(res)}]}

        elif tool_name == "compute_grpo_advantages":
            advs = BUFFER.compute_group_relative_advantages(args["group_trajectories"])
            return {"content": [{"type": "text", "text": json.dumps({"advantages": advs})}]}

        return {"error": f"Unknown tool: {tool_name}"}

    return {"error": f"Unknown method: {method}"}


def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            resp = handle_request(req)
            resp["id"] = req.get("id")
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"error": str(e)}) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
