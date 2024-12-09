import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

# Game Constants
SCREEN_WIDTH = 288
SCREEN_WIDTH_HALF = SCREEN_WIDTH // 2
SCREEN_WIDTH_INV = 1 / SCREEN_WIDTH
SCREEN_HEIGHT = 512
SCREEN_HEIGHT_HALF = SCREEN_HEIGHT // 2
SCREEN_HEIGHT_INV = 1 / SCREEN_HEIGHT
FPS = 60

# Bird Constants
BIRD_SIZE = 20
BIRD_SIZE_HALF = BIRD_SIZE // 2
BIRD_INITIAL_VELOCITY = 0
BIRD_FLAP_VELOCITY = -9
BIRD_GRAVITY = 1
BIRD_X_POS = 50
BIRD_COLOR = (248, 240, 70)

# Pipe Constants
PIPE_WIDTH = 52
PIPE_GAP_SIZE = 150
PIPE_GAP_SIZE_HALF = PIPE_GAP_SIZE // 2
PIPE_COLOR = (104, 200, 79)
PIPE_SPEED = 4
PIPE_MIN_HEIGHT = 150
PIPE_MAX_HEIGHT = 362

# Game Settings
MAX_SCORE = 500
BACKGROUND_COLOR = (135, 206, 235)  # Sky blue

# Reward Constants
ALIVE_REWARD = 1 # reaching 1 score gives 86 rewards for staying alive
PIPE_PASS_REWARD = 86
DEATH_REWARD = -100

class Pipe:
    def __init__(self, x, screen_height):
        self.screen_height = screen_height
        self.reset(x)

    def reset(self, x):
        self.x = x
        self.gap_y = np.random.randint(PIPE_MIN_HEIGHT, PIPE_MAX_HEIGHT)

    def move(self):
        self.x -= PIPE_SPEED
        if self.x < -PIPE_WIDTH:
            self.reset(SCREEN_WIDTH)
            return True
        return False

    def get_rects(self, bird_y):
        upper = pygame.Rect(
            self.x, 
            0, 
            PIPE_WIDTH, 
            self.gap_y - PIPE_GAP_SIZE_HALF
        )
        lower = pygame.Rect(
            self.x, 
            self.gap_y + PIPE_GAP_SIZE_HALF,
            PIPE_WIDTH, 
            self.screen_height - (self.gap_y + PIPE_GAP_SIZE_HALF)
        )
        bird_rect = pygame.Rect(
            BIRD_X_POS, 
            int(bird_y) - BIRD_SIZE_HALF,
            BIRD_SIZE, 
            BIRD_SIZE
        )
        return bird_rect, upper, lower

class FlappyBirdEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super(FlappyBirdEnv, self).__init__()
        # Actions: 0 - do nothing, 1 - flap
        self.action_space = spaces.Discrete(2)
        # Observations: [vertical_distance, horizontal_distance, bird_velocity]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.pipe = Pipe(x=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT)

        self.score = 0
        self.bird_y = SCREEN_HEIGHT_HALF
        self.bird_velocity = BIRD_INITIAL_VELOCITY
        self.isopen = True
        self.screen = None
        self.clock = None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.bird_y = SCREEN_HEIGHT_HALF
        self.bird_velocity = BIRD_INITIAL_VELOCITY
        self.pipe.reset(SCREEN_WIDTH)
        self.score = 0
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        reward = ALIVE_REWARD
        
        if action == 1:
            self.bird_velocity = BIRD_FLAP_VELOCITY
        else:
            self.bird_velocity += BIRD_GRAVITY
        self.bird_y += self.bird_velocity

        passed = self.pipe.move()
        if passed:
            self.score += 1
            reward += PIPE_PASS_REWARD
            if self.score >= MAX_SCORE:
                done = True
                observation = self._get_observation()
                return observation, reward, done, False, {}

        bird_rect, upper_pipe_rect, lower_pipe_rect = self.pipe.get_rects(self.bird_y)
        done = (
            self.bird_y <= 0 or 
            self.bird_y >= SCREEN_HEIGHT or 
            bird_rect.colliderect(upper_pipe_rect) or 
            bird_rect.colliderect(lower_pipe_rect)
        )
        
        if done:
            reward += DEATH_REWARD
            
        observation = self._get_observation()
        info = {}

        return observation, reward, done, False, info

    def _get_observation(self):
        vertical_distance = (self.pipe.gap_y - self.bird_y) * SCREEN_HEIGHT_INV
        horizontal_distance = (self.pipe.x - BIRD_X_POS) * SCREEN_WIDTH_INV
        bird_velocity = self.bird_velocity * 0.1
        return np.array([
            vertical_distance, # Vertical distance to the gap (height difference)
            horizontal_distance, # Horizontal distance to the next pipe
            bird_velocity # Bird's vertical velocity
        ], dtype=np.float32)

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 36)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.isopen = False

        self.screen.fill(BACKGROUND_COLOR)

        bird_rect, upper_pipe_rect, lower_pipe_rect = self.pipe.get_rects(self.bird_y)

        pygame.draw.rect(self.screen, PIPE_COLOR, upper_pipe_rect)
        pygame.draw.rect(self.screen, PIPE_COLOR, lower_pipe_rect)

        pygame.draw.rect(self.screen, BIRD_COLOR, bird_rect)

        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        if self.isopen:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
