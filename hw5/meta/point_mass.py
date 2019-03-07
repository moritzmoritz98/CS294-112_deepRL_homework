import numpy as np
from gym import spaces
from gym import Env


class PointEnv(Env):
    """
    point mass on a 2-D plane
    goals are sampled randomly from a square
    """

    def __init__(self, num_tasks=1, checkerboard_size=0.1):
        self.checkerboard_size = checkerboard_size
        self.reset_task()
        self.reset()
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(2,),
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=-0.1,
                                       high=0.1,
                                       shape=(2,),
                                       dtype=np.float32)
        self._state = None

    def reset_task(self, is_evaluation=False):
        '''
        sample a new task randomly

        Problem 3: make training and evaluation goals disjoint sets
        if `is_evaluation` is true, sample from the evaluation set,
        otherwise sample from the training set
        '''
        # ==================================================================== #
        #                     ----------PROBLEM 3----------
        # ==================================================================== #
        STATIC_SIZE = 10  # final normalized board range: -10..10

        x_cell = np.random.uniform(0, 1)
        y_cell = np.random.uniform(0, 1)
        if self.checkerboard_size == 0:
            # normalize to -10..10
            self._goal = np.array([x_cell, y_cell]) * STATIC_SIZE * 2 - STATIC_SIZE
        else:
            # NB, size of the checkerboard should be quadratic and even to stay fair
            WIDTH = HEIGHT = int(1. / self.checkerboard_size)
            BLACK = int(np.ceil(WIDTH * HEIGHT / 2.))  # evaluation
            WHITE = int(np.floor(WIDTH * HEIGHT / 2.))  # train
            ROW_WIDTH = int(np.ceil(WIDTH / 2))

            if is_evaluation:
                pos = np.random.randint(0, BLACK)
                y_pos = 2 * int(pos / WIDTH) + int((pos % WIDTH) / ROW_WIDTH)
                x_pos = 2 * ((pos % WIDTH) % ROW_WIDTH) + (y_pos % 2)
            else:
                pos = np.random.randint(0, WHITE)
                y_pos = 2 * int(pos / WIDTH) + int((pos % WIDTH) / ROW_WIDTH)
                x_pos = 2 * ((pos % WIDTH) % ROW_WIDTH) + (1 - (y_pos % 2))
            y = y_cell + y_pos
            x = x_cell + x_pos
            # normalize to -10..10
            self._goal = np.array([x, y]) * STATIC_SIZE * 2 / WIDTH - STATIC_SIZE

    def reset(self):
        self._state = np.array([0, 0], dtype=np.float32)
        return self._get_obs()

    def _get_obs(self):
        return np.copy(self._state)

    def reward_function(self, x, y):
        return - (x ** 2 + y ** 2 + 1e-8) ** 0.5

    def step(self, action):
        x, y = self._state
        # compute reward, add penalty for large actions instead of clipping them
        x -= self._goal[0]
        y -= self._goal[1]
        # check if task is complete
        done = abs(x) < .01 and abs(y) < .01
        reward = self.reward_function(x, y)
        # move to next state
        self._state = self._state + action
        ob = self._get_obs()
        return ob, reward, done, dict()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self, mode='ansi'):
        assert mode == 'ansi', 'Error: Human and rgb_array rendering is not supported yet.'
        print('current state:', self._state)

    def seed(self, seed=None):
        assert seed is not None, 'Error: A seed has to be provided.'
        np.random.seed = seed
        return [seed]
