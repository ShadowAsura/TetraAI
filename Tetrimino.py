import pygame
class Tetrimino:
    def __init__(self, shape_data):
        self.shape = shape_data
        self.rotation = 0

    @property
    def current_shape(self):
        return self.shape[self.rotation]

    def rotate(self):
        potential_rotation = (self.rotation + 1) % len(self.shape)
        return self.shape[potential_rotation]

