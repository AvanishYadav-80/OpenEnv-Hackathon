from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple

class Resource(BaseModel):
    id: str = Field(description="Unique ID of the cloud resource")
    instance_type: str = Field(description="Instance type: 'small', 'medium', 'large', 'xlarge', or '2xlarge'")
    cpu_utilization: float = Field(description="Current CPU utilization from 0.0 to 1.0")
    is_critical: bool = Field(description="If true, terminating this resource will cause an outage")
    cost_per_hour: float = Field(description="Hourly cost in USD")

class Observation(BaseModel):
    resources: List[Resource] = Field(description="List of active cloud resources")

class Action(BaseModel):
    resource_id: str = Field(description="ID of the resource to modify")
    operation: str = Field(description="Operation to perform: 'resize', 'terminate', or 'keep'")
    target_type: Optional[str] = Field(None, description="If operation is 'resize', the new instance type")

class Reward(BaseModel):
    score: float = Field(description="Reward signal. Positive for cost saving, negative for outages.")

class CloudOptimizerState(BaseModel):
    resources: Dict[str, Resource]
    processed_resources: set
    initial_total_cost: float
    current_total_cost: float
    crashes: int

INSTANCE_SPECS = {
    "small": {"capacity": 25, "cost": 2.0},
    "medium": {"capacity": 50, "cost": 4.0},
    "large": {"capacity": 100, "cost": 8.0},
    "xlarge": {"capacity": 200, "cost": 16.0},
    "2xlarge": {"capacity": 400, "cost": 32.0}
}

class CloudOptimizerEnv:
    def __init__(self):
        self._state: CloudOptimizerState = None
        self._optimal_savings = 0.0

    def _setup_task(self, task_level: str):
        if task_level == "easy":
            res_list = [
                Resource(id="web-1", instance_type="medium", cpu_utilization=0.0, is_critical=False, cost_per_hour=4.0)
            ]
            self._optimal_savings = 4.0 # can terminate
        elif task_level == "medium":
            res_list = [
                Resource(id="db-master", instance_type="xlarge", cpu_utilization=0.20, is_critical=True, cost_per_hour=16.0),
                Resource(id="cache-node", instance_type="large", cpu_utilization=0.95, is_critical=True, cost_per_hour=8.0),
                Resource(id="worker-idle", instance_type="small", cpu_utilization=0.05, is_critical=False, cost_per_hour=2.0),
                Resource(id="api-gateway", instance_type="2xlarge", cpu_utilization=0.10, is_critical=True, cost_per_hour=32.0)
            ]
            # db-master: 200 * 0.20 = 40 units -> resize to medium (capacity 50), new cost 4. Savings = 12
            # cache-node: keep. Savings = 0
            # worker-idle: terminate. Savings = 2
            # api-gateway: 400 * 0.10 = 40 units -> resize to medium (capacity 50), new cost 4. Savings = 28
            # Total optimal savings = 12 + 0 + 2 + 28 = 42
            self._optimal_savings = 42.0

        elif task_level == "hard":
            res_list = [
                Resource(id="queue-1", instance_type="large", cpu_utilization=0.45, is_critical=True, cost_per_hour=8.0),
                Resource(id="search-1", instance_type="2xlarge", cpu_utilization=0.22, is_critical=True, cost_per_hour=32.0),
                Resource(id="search-2", instance_type="2xlarge", cpu_utilization=0.22, is_critical=True, cost_per_hour=32.0),
                Resource(id="analytics", instance_type="xlarge", cpu_utilization=0.80, is_critical=False, cost_per_hour=16.0),
                Resource(id="staging-db", instance_type="xlarge", cpu_utilization=0.0, is_critical=False, cost_per_hour=16.0),
                Resource(id="ml-inference", instance_type="large", cpu_utilization=0.55, is_critical=True, cost_per_hour=8.0),
                Resource(id="proxy-1", instance_type="small", cpu_utilization=0.99, is_critical=True, cost_per_hour=2.0),
                Resource(id="log-shipper", instance_type="medium", cpu_utilization=0.10, is_critical=False, cost_per_hour=4.0)
            ]
            # q1: 100 * .45 = 45 -> medium (50). cost 4. Save 4
            # s1: 400 * .22 = 88 -> large (100). cost 8. Save 24
            # s2: 400 * .22 = 88 -> large (100). cost 8. Save 24
            # analytics: 200 * .8 = 160 -> keep. Save 0
            # staging-db: term -> Save 16
            # ml-inf: 100 * .55 = 55 -> keep (cannot go medium since 55>50). Save 0
            # proxy-1: small (25) -> keep. Save 0 
            # log-shipper: medium 50*.1 = 5 -> term. Save 4
            # Total optimal = 4 + 24 + 24 + 0 + 16 + 0 + 0 + 4 = 72
            self._optimal_savings = 72.0
        else:
            raise ValueError("task_level must be 'easy', 'medium', or 'hard'")
        
        initial_cost = sum(r.cost_per_hour for r in res_list)
        self._state = CloudOptimizerState(
            resources={r.id: r for r in res_list},
            processed_resources=set(),
            initial_total_cost=initial_cost,
            current_total_cost=initial_cost,
            crashes=0
        )

    def reset(self, task_level: str = "medium") -> Observation:
        self._setup_task(task_level)
        return self._obs()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, dict]:
        if not self._state:
            raise RuntimeError("Environment must be reset before calling step.")

        if action.resource_id not in self._state.resources or action.resource_id in self._state.processed_resources:
            return self._obs(), Reward(score=-0.1), False, {"error": "Invalid or already processed resource_id"}

        res = self._state.resources[action.resource_id]
        step_reward = 0.0
        message = "Kept as is."

        if action.operation == "terminate":
            if res.is_critical:
                self._state.crashes += 1
                step_reward = -1.0
                message = f"CRASH: Terminated critical resource {res.id}!"
            else:
                saved = res.cost_per_hour
                step_reward = (saved / self._optimal_savings) if self._optimal_savings else 1.0
                self._state.current_total_cost -= saved
                message = f"Successfully terminated. Saved ${saved}/hr."
                
        elif action.operation == "resize":
            if action.target_type not in INSTANCE_SPECS:
                return self._obs(), Reward(score=-0.1), False, {"error": "Invalid target_type"}
            
            old_spec = INSTANCE_SPECS[res.instance_type]
            new_spec = INSTANCE_SPECS[action.target_type]
            
            units_used = old_spec["capacity"] * res.cpu_utilization
            new_utilization = units_used / new_spec["capacity"] # e.g. 40 / 50 = 0.8
            
            if new_utilization > 1.0:
                # OOM or CPU saturation crash
                self._state.crashes += 1
                step_reward = -1.0
                message = f"CRASH: Resized {res.id} to {action.target_type} but needed {units_used} units (Capacity {new_spec['capacity']})."
            else:
                saved = old_spec["cost"] - new_spec["cost"]
                # Only reward positive savings
                if saved > 0:
                    step_reward = saved / self._optimal_savings
                    self._state.current_total_cost -= saved
                    message = f"Resized to {action.target_type}. New Util: {new_utilization:.0%}. Saved ${saved}/hr."
                elif saved < 0:
                    step_reward = -0.2
                    self._state.current_total_cost -= saved
                    message = f"Upsized unnecessarily. Wasted ${-saved}/hr."
                else: # saved == 0
                    message = "Resized to same capacity. No savings."
                    
        elif action.operation == "keep":
            step_reward = 0.0
            message = "Resource maintained safely."

        self._state.processed_resources.add(res.id)
        
        info = {
            "message": message,
            "step_reward": step_reward,
            "current_cost": self._state.current_total_cost,
            "crashes": self._state.crashes
        }

        done = len(self._state.processed_resources) == len(self._state.resources)
        
        if done:
            savings = self._state.initial_total_cost - self._state.current_total_cost
            if self._state.crashes > 0:
                final_score = 0.01 # Any crash means total failure, strictly > 0
            else:
                raw_score = savings / self._optimal_savings if self._optimal_savings else 0.99
                final_score = max(0.01, min(0.99, raw_score))
            info["final_score"] = final_score

        return self._obs(), Reward(score=step_reward), done, info

    def state(self) -> CloudOptimizerState:
        return self._state

    def _obs(self) -> Observation:
        unprocessed = [r for r in self._state.resources.values() if r.id not in self._state.processed_resources]
        return Observation(resources=unprocessed)
