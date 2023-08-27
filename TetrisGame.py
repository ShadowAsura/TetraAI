import pygame
import random

# Initialize pygame
pygame.init()

# Colors and constants
WHITE = (255, 255, 255)
GRID_SIZE = (10, 20)
BLOCK_SIZE = 30
SCREEN_SIZE = (GRID_SIZE[0] * BLOCK_SIZE, GRID_SIZE[1] * BLOCK_SIZE)
FPS = 60

# Tetriminos shapes and rotations
TETRIMINOS = {
    "I": [
        [(0, 1), (1, 1), (2, 1), (3, 1)],
        [(2, 0), (2, 1), (2, 2), (2, 3)]
    ],
    "O": [
        [(1, 1), (1, 2), (2, 1), (2, 2)]
    ],
    "T": [
        [(1, 1), (0, 2), (1, 2), (2, 2)],
        [(1, 1), (0, 2), (1, 2), (1, 3)],
        [(1, 2), (0, 2), (1, 1), (2, 2)],
        [(1, 2), (1, 1), (1, 3), (2, 2)]
    ],
    # ... (Other Tetriminos)
}


class TetrisGame:
    def __init__(self):
        # ... (Previous code)
        self.grid = [[0 for _ in range(GRID_SIZE[0])] for _ in range(GRID_SIZE[1])]
        self.current_tetrimino = random.choice(list(TETRIMINOS.keys()))
        self.current_position = (0, 0)
        self.next_tetrimino = random.choice(list(TETRIMINOS.keys()))
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
        # Get the next rotation
        current_rotation = TETRIMINOS[self.current_tetrimino]
        next_rotation = current_rotation[(current_rotation.index(self.current_tetrimino) + 1) % len(current_rotation)]
        if not self.check_collision(self.current_position, next_rotation):
            self.current_tetrimino = next_rotation

    def spawn_tetrimino(self):
        self.current_tetrimino = self.next_tetrimino
        self.next_tetrimino = random.choice(list(TETRIMINOS.keys()))
        self.current_position = (GRID_SIZE[0] // 2, 0)
        if self.check_collision(self.current_position, TETRIMINOS[self.current_tetrimino][0]):
            # Game over if collision at spawn
            self.reset()

    def drop(self):
        while not self.check_collision((self.current_position[0], self.current_position[1] + 1), TETRIMINOS[self.current_tetrimino][0]):
            self.current_position = (self.current_position[0], self.current_position[1] + 1)
        self.place_tetrimino()


    def check_collision(self, position, tetrimino):
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

    def draw_tetrimino(self, position, tetrimino):
        for x, y in tetrimino:
            pygame.draw.rect(self.screen, (0, 128, 255), ((x + position[0]) * BLOCK_SIZE, (y + position[1]) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

    def place_tetrimino(self):
        for x, y in TETRIMINOS[self.current_tetrimino][0]:
            self.grid[y + self.current_position[1]][x + self.current_position[0]] = 1
        self.clear_lines()
        self.spawn_tetrimino()

    def clear_lines(self):
        lines_to_clear = [y for y in range(GRID_SIZE[1]) if all(self.grid[y])]
        for y in lines_to_clear:
            del self.grid[y]
            self.grid.insert(0, [0 for _ in range(GRID_SIZE[0])])


    def game_over(self):
        # Check if the game is over
        pass

    def run(self):
        drop_counter = 0
        drop_speed = 500  # milliseconds
        last_drop_time = pygame.time.get_ticks()

        running = True
        while running:
            current_time = pygame.time.get_ticks()
            if current_time - last_drop_time > drop_speed:
                if not self.check_collision((self.current_position[0], self.current_position[1] + 1), TETRIMINOS[self.current_tetrimino][0]):
                    self.current_position = (self.current_position[0], self.current_position[1] + 1)
                else:
                    self.place_tetrimino()
                last_drop_time = current_time

            for event in pygame.event.get():
                pass
            self.screen.fill(WHITE)
            self.draw_grid()
            self.draw_tetrimino(self.current_position, TETRIMINOS[self.current_tetrimino][0])

            pygame.display.flip()
            self.clock.tick(FPS)



if __name__ == "__main__":
    game = TetrisGame()
    game.run()
