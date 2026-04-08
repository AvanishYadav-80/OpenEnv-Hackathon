import os
import json
from env import CloudOptimizerEnv, Action, INSTANCE_SPECS
from openai import OpenAI

# Required by Hackathon Pre-Submission Checklist
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL
)

BENCHMARK = "cloud-cost-optimizer"

def run_inference(task_level: str):
    print(f"[START] task={task_level} env={BENCHMARK} model={MODEL_NAME}")
    
    env = CloudOptimizerEnv()
    obs = env.reset(task_level)
    
    done = False
    step_num = 1
    rewards_history = []
    
    while not done:
        system_prompt = f"""You are a Cloud Infrastructure SRE optimizing costs.
You must process ONE unprocessed resource from the observation.

Instance capacities and costs:
{json.dumps(INSTANCE_SPECS, indent=2)}

Rules:
1. Target `target_type` must be one of the instance keys above.
2. Capacity constraint: If (old_capacity * cpu_utilization) > new_capacity, the node crashes. DO NOT downsize if it exceeds 100%.
3. Do not `terminate` a resource if `is_critical` is true.

Output your action in JSON exactly matching this schema:
{{
    "resource_id": "<string>",
    "operation": "<terminate | resize | keep>",
    "target_type": "<string | null>"
}}"""
        
        user_prompt = f"Unprocessed resources to evaluate:\n{obs.model_dump_json(indent=2)}\nProvide your JSON action for ONE resource:"

        action_str = "{}"
        error_str = "null"
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            raw_action = response.choices[0].message.content
            action_data = json.loads(raw_action)
            action_str = json.dumps(action_data, separators=(',', ':'))
            action = Action(**action_data)
            obs, reward, done, info = env.step(action)
            
            error_msg = info.get("error", None)
            if error_msg:
                error_str = f"\"{error_msg}\""
                
        except Exception as e:
            error_str = f"\"{str(e)}\""
            reward = type('Reward', (), {'score': 0.0})()
            obs, reward_obj, done, info = env.reset(task_level), type('Reward', (), {'score': 0.0})(), True, {"final_score": 0.01}

        rewards_history.append(reward.score)
        done_val = "true" if done else "false"
        
        # [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
        print(f"[STEP] step={step_num} action={action_str} reward={reward.score:.2f} done={done_val} error={error_str}")
        
        step_num += 1
    
    score = info.get('final_score', 0.01)
    success_val = "true" if score > 0.01 else "false"
    rewards_str = ",".join([f"{r:.2f}" for r in rewards_history])
    
    # [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
    print(f"[END] success={success_val} steps={step_num-1} score={score:.2f} rewards={rewards_str}")

if __name__ == "__main__":
    for level in ["easy", "medium", "hard"]:
        run_inference(level)
