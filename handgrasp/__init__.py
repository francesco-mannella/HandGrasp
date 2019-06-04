from gym.envs.registration import register

register(id='HandGrasp-v0', 
    entry_point='handgrasp.envs:HandGraspEnv', 
)

from handgrasp.envs import handgrasp_env
