import pygame
class Tetrimino:
    def __init__(self, shape):
        self.shape = shape
        self.rotation = 0

    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.shape)
        return self.shape[self.rotation]