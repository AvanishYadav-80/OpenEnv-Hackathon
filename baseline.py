import os
import json
from env import CloudOptimizerEnv, Action, INSTANCE_SPECS
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key-for-test"))

def run_baseline(task_level: str):
    print(f"\n{'='*50}\nTask Level: {task_level.upper()}\n{'='*50}")
    env = CloudOptimizerEnv()
    obs = env.reset(task_level)
    
    done = False
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

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            raw_action = response.choices[0].message.content
            action_data = json.loads(raw_action)
            action = Action(**action_data)
        except Exception as e:
            print(f"Failed API or parsing: {e}")
            break

        obs, reward, done, info = env.step(action)
        print(f"Action -> ID: {action.resource_id} | OP: {action.operation} | TARGET: {action.target_type}")
        print(f"Result -> {info['message']} | Step Reward: {reward.score:.2f}")
    
    if done:
        score = info.get('final_score', 0)
        crashes = info.get('crashes', 0)
        print(f"\n>>> Final Grader Score: {score:.2f} (0.0=Fail, 1.0=Optimal Savings). Crashes: {crashes}")

if __name__ == "__main__":
    if os.getenv("OPENAI_API_KEY") is None:
        print("WARNING: OPENAI_API_KEY environment variable not set. The API call will fail unless a mock is used.")
    
    for level in ["easy", "medium", "hard"]:
        run_baseline(level)
