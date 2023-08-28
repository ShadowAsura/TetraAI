import pygame
from copy import deepcopy

def simulate_move(game_state, tetrimino, move):
    new_state = deepcopy(game_state)
    if move == "left":
        # Shift tetrimino left
        pass
    elif move == "right":
        # Shift tetrimino right
        pass
    elif move == "down":
        # Shift tetrimino down
        pass
    elif move == "rotate":
        # Rotate tetrimino
        pass
    return new_state


def simulate_drop(game_state, tetrimino):
    new_state = deepcopy(game_state)
    # Drop the tetrimino to the lowest possible position
    pass
    return new_state

def evaluate_state(state):
    score = 0
    
    # Check for line clears
    for row in state:
        if sum(row) == len(row):  # full row
            score += 10

    # Check for holes
    for col in range(len(state[0])):
        column_data = [row[col] for row in state]
        if 1 in column_data:
            top_block_index = column_data.index(1)
            score -= 5 * column_data[top_block_index:].count(0)

    # Check height
    highest_block = 20
    for row in state:
        if 1 in row:
            highest_block = state.index(row)
            break
    score -= highest_block

    return score

def possible_moves(tetrimino):
    return ["left", "right", "down", "rotate"]

def best_move(game_state, tetrimino):
    best_score = float('-inf')
    best_move = None

    for move in possible_moves(tetrimino):
        new_state = simulate_move(game_state, tetrimino, move)
        score = evaluate_state(new_state)
        if score > best_score:
            best_score = score
            best_move = move

    return best_move
