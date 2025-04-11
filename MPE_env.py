from environment import MultiAgentEnv
from simple_spread import Scenario

def MPEEnv(args):
    
    scenario = Scenario()
    world = scenario.make_world(args)
    
    env = MultiAgentEnv(world, scenario.reset_world,
                        scenario.reward, scenario.observation, scenario.info)

    return env
