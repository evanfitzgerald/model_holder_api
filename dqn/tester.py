
import numpy as np
from gym import spaces

x_threshold = 2.4

# Angle limit set to 2 * theta_threshold_radians so failing observation
# is still within bounds.
high = np.array(
    [
        x_threshold * 2
    ],
    dtype=np.float32,
)

action_space = spaces.Discrete(2)
observation_space = spaces.Box(-high, high, dtype=np.float32)

print(observation_space.shape[0])