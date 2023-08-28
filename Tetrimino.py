import pygame
import random

colors = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (255, 165, 0)  # Orange
]
def get_random_color():
    return random.choice(colors)

class Tetrimino:
    def __init__(self, shape_data):
        self.shape = shape_data
        self.rotation = 0
        self.color = get_random_color()

    @property
    def current_shape(self):
        return self.shape[self.rotation]

    def rotate(self):
        potential_rotation = (self.rotation + 1) % len(self.shape)
        return self.shape[potential_rotation]



