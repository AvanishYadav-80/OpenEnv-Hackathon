from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from env import CloudOptimizerEnv, Action

app = FastAPI(title="OpenEnv Cloud Cost Optimizer")

env_instance = CloudOptimizerEnv()

class ResetRequest(BaseModel):
    task_level: str = "medium"

@app.post("/reset")
def reset_env(req: ResetRequest):
    obs = env_instance.reset(req.task_level)
    return {"observation": obs.model_dump()}

@app.post("/step")
def step_env(action: Action):
    try:
        obs, reward, done, info = env_instance.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def get_state():
    if not env_instance._state:
        return {"error": "Environment not initialized. Call /reset first."}
    return env_instance.state().model_dump()
