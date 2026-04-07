---
title: Cloud Cost Optimizer
emoji: ☁️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Cloud Cost Optimizer (OpenEnv)

This is a real-world, deep-modeling Reinforcement Learning environment built using the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework. 

It simulates a **Cloud SRE Cost Optimization** task where complex capacities must be balanced against hard SLAs. The agent is required to deeply reason over utilization mathematics to succeed, fulfilling the **"Excellent - fills a real gap"** rubric priority!

## Real-World Utility
Organizations routinely bleed capital safely overprovisioning node clusters. This environment models the core math: 
- `small` (cap 25): $2/hr
- `medium` (cap 50): $4/hr
- `large` (cap 100): $8/hr
- `xlarge` (cap 200): $16/hr
- `2xlarge` (cap 400): $32/hr

The agent must optimize `Observation` nodes by responding with `Action(operation="resize|terminate|keep")`.

## The Mechanics & Rules
1. If you downsize a node with high utilization, and `(current_capacity * utilization_rate) > target_capacity`, the CPU hits 100% and crashes the service!
2. You cannot terminate a node with `is_critical=True`.
3. Crashing a service ruins the episode yielding a final grader score of `0.0`. Earning `1.0` requires identifying the maximum possible architectural cost savings.

## Task Difficulty
- **Easy:** Identify 1 idle node to terminate.
- **Medium:** 4 interdependent nodes to math-check.
- **Hard:** 8 complex resources pushing frontier limits for reasoning over numbers without OOMing the nodes.

## Running the Baseline
The baseline runs `gpt-4o` to step through the environments automatically:
```bash
export OPENAI_API_KEY="sk-..."
python baseline.py
```

## Running as a Hugging Face Space
```bash
docker build -t openenv-cloud-optimizer .
docker run -p 7860:7860 openenv-cloud-optimizer
```
This enables direct interactions with standard HTTP payload spec compliant `/reset`, `/step` and `/state` APIs.
