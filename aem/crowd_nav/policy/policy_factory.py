from aem.crowd_sim.envs.policy.policy_factory import policy_factory
from aem.crowd_nav.policy.cadrl import CADRL
from aem.crowd_nav.policy.lstm_rl import LstmRL
from aem.crowd_nav.policy.sarl import SARL
from aem.crowd_nav.policy.comcarl import COMCARL
from aem.crowd_nav.policy.gipcarl import GIPCARL
from aem.crowd_nav.policy.actcarl import ACTCARL
from aem.crowd_nav.policy.actenvcarl import ACTENVCARL
from aem.crowd_nav.policy.actfcarl import ACTFCARL


policy_factory['cadrl'] = CADRL
policy_factory['lstm_rl'] = LstmRL
policy_factory['sarl'] = SARL
policy_factory['comcarl'] = COMCARL
policy_factory['gipcarl'] = GIPCARL
policy_factory['actcarl'] = ACTCARL
policy_factory['actenvcarl'] = ACTENVCARL
policy_factory['actfcarl'] = ACTFCARL
