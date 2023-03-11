from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState


class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        # print(f'state: {state}')
        action = self.policy.predict(state)
        # print("self.policy.last_state: ",self.policy.last_state)
        return action
