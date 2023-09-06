import pygame
import random
import sys
import numpy as np
from Tetrimino import Tetrimino
from q_network import QNetwork
from replay_buffer import ReplayBuffer

# Initialize pygame
pygame.init()

# Colors and constants
WHITE = (255, 255, 255)
GRID_SIZE = (10, 20)
BLOCK_SIZE = 30
SCREEN_SIZE = (GRID_SIZE[0] * BLOCK_SIZE + 150, GRID_SIZE[1] * BLOCK_SIZE)  # +150 pixels for the preview box

FPS = 60

# Initialize network and hyperparameters
input_dim = 200
hidden_dim = 128
output_dim = 4
q_network = QNetwork(input_dim, hidden_dim, output_dim)
replay_buffer = ReplayBuffer(10000)

GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
EPISODES = 1000


# Tetriminos shapes and rotations
TETRIMINOS = {
    "I": [
        [(0, 1), (1, 1), (2, 1), (3, 1)],
        [(2, 0), (2, 1), (2, 2), (2, 3)]
    ],
    "J": [
        [(0, 1), (0, 2), (1, 2), (2, 2)],
        [(1, 0), (2, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1), (2, 0)],
        [(1, 0), (1, 1), (1, 2), (2, 2)]
    ],
    "L": [
        [(0, 1), (1, 1), (2, 1), (2, 2)],
        [(1, 0), (1, 1), (1, 2), (2, 0)],
        [(0, 1), (1, 1), (2, 1), (0, 0)],
        [(1, 0), (1, 1), (1, 2), (0, 2)]
    ],
    "O": [
        [(1, 1), (1, 2), (2, 1), (2, 2)]
    ],
    "S": [
        [(1, 1), (2, 1), (0, 2), (1, 2)],
        [(1, 0), (1, 1), (2, 1), (2, 2)]
    ],
    "Z": [
        [(0, 1), (1, 1), (1, 2), (2, 2)],
        [(2, 0), (1, 1), (2, 1), (1, 2)]
    ],
    "T": [
        [(1, 1), (0, 2), (1, 2), (2, 2)],
        [(1, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1), (1, 2)],
        [(1, 0), (1, 1), (1, 2), (2, 1)]
    ]
}



class TetrisGame:
    def __init__(self):
        self.grid = [[0 for _ in range(GRID_SIZE[0])] for _ in range(GRID_SIZE[1])]
        
        # Initialize with a Tetrimino instance
        self.next_tetrimino = random.choice(list(TETRIMINOS.keys()))
        self.spawn_tetrimino()  # This will set self.current_tetrimino
        self.current_position = (GRID_SIZE[0] // 2, 0)  # Initialize the current_position
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        self.clock = pygame.time.Clock()




    def reset(self):
        self.grid = [[0 for _ in range(GRID_SIZE[0])] for _ in range(GRID_SIZE[1])]
        self.spawn_tetrimino()


    def move(self, direction):
        dx, dy = direction
        new_position = (self.current_position[0] + dx, self.current_position[1] + dy)
        if not self.check_collision(new_position, self.current_tetrimino):
            self.current_position = new_position

    def rotate(self):
        # Get the potential next rotated shape
        potential_rotated_shape = self.current_tetrimino.rotate()

        # If the new rotation does not result in a collision, update the tetrimino's rotation
        if not self.check_collision(self.current_position, potential_rotated_shape):
            self.current_tetrimino.rotation += 1
            self.current_tetrimino.rotation %= len(self.current_tetrimino.shape)
        else:
            # If it does result in a collision, revert the rotation to its previous state
            self.current_tetrimino.rotation -= 1
            self.current_tetrimino.rotation %= len(self.current_tetrimino.shape)

    def draw_preview_box(self):
        pygame.draw.rect(self.screen, (0, 0, 0), (GRID_SIZE[0] * BLOCK_SIZE, 0, 150, 5 * BLOCK_SIZE), 2)  # Draws a rectangle with a 2 pixel border width
        pygame.draw.rect(self.screen, (220, 220, 220), (GRID_SIZE[0] * BLOCK_SIZE + 2, 2, 146, 5 * BLOCK_SIZE - 4))  # Fills the inside of the rectangle


    def spawn_tetrimino(self):
        self.current_tetrimino = Tetrimino(TETRIMINOS[self.next_tetrimino])

        self.next_tetrimino = random.choice(list(TETRIMINOS.keys()))
        self.current_position = (GRID_SIZE[0] // 2, 0)
        if self.check_collision(self.current_position, self.current_tetrimino.current_shape):
            # Game over if collision at spawn
            self.reset()


    def drop(self):
        while not self.check_collision((self.current_position[0], self.current_position[1] + 1), TETRIMINOS[self.current_tetrimino][0]):
            self.current_position = (self.current_position[0], self.current_position[1] + 1)
        self.place_tetrimino()


    def check_collision(self, position, tetrimino):
        print(f"Checking collision with: {tetrimino}")
        for x, y in tetrimino:
            if x + position[0] >= GRID_SIZE[0] or y + position[1] >= GRID_SIZE[1] or \
               x + position[0] < 0 or y + position[1] < 0 or \
               self.grid[y + position[1]][x + position[0]] == 1:
                return True
        return False

    def draw_grid(self):
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(self.screen, (0, 128, 255), (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        """start_point = (GRID_SIZE[0] * 10, 128)
        end_point = (GRID_SIZE[0], GRID_SIZE[1])
        pygame.draw.line(self.screen, (0,0,0), start_point, end_point, 2)  # Using white color for the line"""

    def draw_tetrimino(self, position, tetrimino):
        if not isinstance(tetrimino, list):
            return
        for x, y in tetrimino:
            pygame.draw.rect(self.screen, self.current_tetrimino.color, ((x + position[0]) * BLOCK_SIZE, (y + position[1]) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

    def draw_next_tetrimino(self):
        tetrimino_shape = TETRIMINOS[self.next_tetrimino][0]  # Get the first rotation of the next tetrimino
        middle_preview_box = 2.5 * BLOCK_SIZE  # This is half of 5 * BLOCK_SIZE which is our height for preview box

        
        # Calculate the width and height of the tetrimino
        min_x = min([x for x, y in tetrimino_shape])
        max_x = max([x for x, y in tetrimino_shape])
        min_y = min([y for x, y in tetrimino_shape])
        max_y = max([y for x, y in tetrimino_shape])
        width = (max_x - min_x + 1) * BLOCK_SIZE
        height = (max_y - min_y + 1) * BLOCK_SIZE

        middle_tetrimino = (max_y + min_y) * BLOCK_SIZE / 2

        # Calculate the offsets to center the tetrimino in the preview box
        offset_x = GRID_SIZE[0] * BLOCK_SIZE + (150 - width) // 2
        offset_y = int(middle_preview_box - middle_tetrimino)

        for x, y in tetrimino_shape:
            pygame.draw.rect(self.screen, (0, 128, 255), ((x * BLOCK_SIZE) + offset_x, (y * BLOCK_SIZE) + offset_y, BLOCK_SIZE, BLOCK_SIZE))



    def place_tetrimino(self):
        for x, y in self.current_tetrimino.current_shape:
            self.grid[y + self.current_position[1]][x + self.current_position[0]] = 1
        self.clear_lines()
        self.spawn_tetrimino()

    def clear_lines(self):
        lines_to_clear = [y for y in range(GRID_SIZE[1]) if all(self.grid[y])]
        for y in lines_to_clear:
            del self.grid[y]
            self.grid.insert(0, [0 for _ in range(GRID_SIZE[0])])


    def move_left(self):
        """Move the current tetrimino one unit to the left if possible."""
        new_position = (self.current_position[0] - 1, self.current_position[1])
        
        # Check if the move is valid
        if not self.check_collision(new_position, self.current_tetrimino.current_shape):
            self.current_position = new_position

    def move_right(self):
        """Move the current tetrimino one unit to the right if possible."""
        new_position = (self.current_position[0] + 1, self.current_position[1])
        
        # Check if the move is valid
        if not self.check_collision(new_position, self.current_tetrimino.current_shape):
            self.current_position = new_position

    def game_over(self):
        """Check if the game is over."""
        # Check if the top row of the grid has any filled cells
        return any(cell == 1 for cell in self.grid[0])

    def get_state(self):
        return self.grid  # assuming this represents your game state

    def step(self, action):
        """Take an action and return the resulting state, reward, and whether the game is done."""
        
        # Map integer actions to string actions if necessary
        if isinstance(action, int):
            action_map = {
                0: "LEFT",
                1: "RIGHT",
                2: "ROTATE",
                3: "DROP"
            }
            action = action_map.get(action, "INVALID")
        
        print(f"Action type: {type(action)}, Action value: {action}")
        print("Received action:", action)

        # Check if the action is a string
        if not isinstance(action, str):
            print(f"Invalid action type: {type(action)}")
            raise ValueError("Invalid action type provided")
        
        if action == "LEFT":
            self.move_left()
        elif action == "RIGHT":
            self.move_right()
        elif action == "ROTATE":
            self.rotate()
        elif action == "DROP":
            self.drop()
        else:
            raise ValueError("Invalid action provided")
        
        reward = self.calculate_reward()  # Here we call the reward function
        next_state = self.get_state()
        done = self.game_over()
        
        return next_state, reward, done


    def run(self):
        global epsilon
        drop_counter = 0
        drop_speed = 500
        last_drop_time = pygame.time.get_ticks()
        
        ai_enabled = True

        running = True
        while running:
            self.clock.tick(FPS)  # Regulate speed
            current_time = pygame.time.get_ticks()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.move_left()
                    elif event.key == pygame.K_RIGHT:
                        self.move_right()
                    elif event.key == pygame.K_UP:
                        self.rotate()
                    elif event.key == pygame.K_DOWN:
                        self.drop()

            if ai_enabled:
                if random.random() < epsilon:  # epsilon-greedy policy
                    move = random.choice(["LEFT", "RIGHT", "ROTATE", "DROP"])
                else:
                    move = best_move(self.get_state(), self.current_tetrimino)
                print("AI generated move:", move)
                next_state, reward, done = self.step(move)

            self.screen.fill(WHITE)
            self.draw_grid()
            self.draw_tetrimino(self.current_position, self.current_tetrimino.current_shape)
            self.draw_next_tetrimino()
            self.draw_preview_box()
            pygame.display.update()
    def calculate_reward(self):
        """
        Calculate reward based on game state.
        """
        reward = 0

        # Check for cleared lines and add reward accordingly
        lines_cleared = len([y for y in range(GRID_SIZE[1]) if all(self.grid[y])])
        if lines_cleared > 0:
            reward += lines_cleared * 10

        # Check for potential collisions for a downward move (bad scenario)
        if self.check_collision((self.current_position[0], self.current_position[1] + 1), self.current_tetrimino.current_shape):
            reward -= 5

        # If the game is over, then heavily penalize the action
        if self.game_over():
            reward -= 100
        
        return reward




# Initialize Tetris game and Q-network
tetris = TetrisGame()


# Main loop
for episode in range(1, EPISODES + 1):
    state = tetris.reset()
    done = False
    episode_reward = 0

    while not done:
        # Choose an action with epsilon-greedy strategy
        if np.random.rand() < EPSILON:
            action = random.randint(0, 3)  # Assuming 4 possible actions
        else:
            action = np.argmax(q_network.predict(state))

        # Take a step
        print(f"Action type: {type(action)}, Action value: {action}")

        next_state, reward, done = tetris.step(action)

        # Update Q-values
        next_state_array = np.array(next_state)
        q_values_next, _ = q_network.forward_pass(next_state_array)
        target = reward + GAMMA * np.max(q_values_next)


        q_network.update(state, action, target)

        # Update state and episode reward
        state = next_state
        episode_reward += reward

    # Decay epsilon
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY

    print(f"Episode: {episode}, Total Reward: {episode_reward}")

# Save the trained model
q_network.save_model("tetris_qnetwork.h5")
