import pygame
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
import sys
import time
import os
from collections import deque

# Suppress Pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

# --- Configuration ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRID_SIZE = 20
MAZE_WIDTH = SCREEN_WIDTH // GRID_SIZE
MAZE_HEIGHT = SCREEN_HEIGHT // GRID_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (100, 100, 100)
LIGHT_BLUE = (173, 216, 230)
GOLD = (255, 215, 0)
PURPLE = (128, 0, 128)
DARK_GREEN = (0, 100, 0)

# UI Colors
UI_BG = (45, 45, 45)
UI_TEXT = (240, 240, 240)
UI_HIGHLIGHT = (255, 255, 0)

# Maze elements
EMPTY = 0
WALL = 1
START = 2
GOAL = 3
PLAYER = 4
PATH = 5

# Default RL Hyperparameters
DEFAULT_EPISODES = 5000
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_DISCOUNT_FACTOR = 0.95
DEFAULT_EPSILON = 1.0
DEFAULT_EPSILON_DECAY = 0.999
DEFAULT_MIN_EPSILON = 0.01
DEFAULT_MAX_STEPS_PER_EPISODE = 200
DEFAULT_RENDER_EVERY = 500

# --- Helper Functions ---
def check_maze_solvability(maze, start_pos, goal_pos):
    """Use BFS to check if there is a path from start to goal"""
    if not start_pos or not goal_pos:
        return False
    
    visited = set()
    queue = deque([start_pos])
    visited.add(start_pos)
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        current = queue.popleft()
        
        if current == goal_pos:
            return True
        
        r, c = current
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < maze.shape[0] and 0 <= nc < maze.shape[1] and
                maze[nr, nc] != WALL and (nr, nc) not in visited):
                queue.append((nr, nc))
                visited.add((nr, nc))
    
    return False

def find_shortest_path(maze, start_pos, goal_pos):
    """Find shortest path using BFS and return the path"""
    if not start_pos or not goal_pos:
        return []
    
    visited = set()
    queue = deque([(start_pos, [])])
    visited.add(start_pos)
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        current, path = queue.popleft()
        
        if current == goal_pos:
            return path + [current]
        
        r, c = current
        for i, (dr, dc) in enumerate(directions):
            nr, nc = r + dr, c + dc
            next_pos = (nr, nc)
            if (0 <= nr < maze.shape[0] and 0 <= nc < maze.shape[1] and
                maze[nr, nc] != WALL and next_pos not in visited):
                queue.append((next_pos, path + [current]))
                visited.add(next_pos)
    
    return []

# --- Visual Elements ---
class Button:
    def __init__(self, x, y, width, height, text, color=UI_BG, hover_color=UI_HIGHLIGHT, text_color=UI_TEXT):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.is_hovered = False
        
    def draw(self, screen, font):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, WHITE, self.rect, 2)
        
        text_surf = font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)
        
    def check_hover(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        return self.is_hovered
        
    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.is_hovered
        return False

# --- Custom Pygame Environment ---
class MazeEnv:
    def __init__(self, maze=None, render_mode='human', max_steps=DEFAULT_MAX_STEPS_PER_EPISODE, 
                 maze_type='static', generation_method='dfs'):
        pygame.init()
        self.width = MAZE_WIDTH
        self.height = MAZE_HEIGHT
        self.grid_size = GRID_SIZE
        self.max_steps = max_steps
        self.current_step = 0
        self.maze_type = maze_type
        self.generation_method = generation_method
        self.game_mode = 'ai'
        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT
        self.ui_font = None
        self.ui_title_font = None
        self.show_path = False
        self.ai_path = []
        self.player_pos = None
        self.buttons = {}
        self.player_moves = 0
        self.player_finish_time = 0
        self.ai_finish_time = 0
        self.race_started = False

        # Initialize fonts
        pygame.font.init()
        self.ui_font = pygame.font.SysFont('Arial', 18)
        self.ui_title_font = pygame.font.SysFont('Arial', 24, bold=True)

        # Generate initial maze
        self.maze = self._generate_solvable_maze()
        self.start_pos = self._find_pos(START)
        self.goal_pos = self._find_pos(GOAL)

        self.agent_pos = self.start_pos
        if self.game_mode == 'player' or self.game_mode == 'race':
            self.player_pos = self.start_pos

        self.action_space = [0, 1, 2, 3]
        self.n_actions = len(self.action_space)
        self.observation_space_shape = (self.height, self.width)

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.background_img = None
        
        self.sprites = {
            'wall': None,
            'floor': None,
            'goal': None,
            'start': None,
            'agent': None,
            'player': None,
            'path': None
        }
        
        if self.render_mode == 'human':
            self._init_render()
            self._load_sprites()
            self._create_ui_elements()

    def _generate_solvable_maze(self):
        """Generate a maze that is guaranteed to be solvable"""
        while True:
            maze = self._generate_maze()
            start_pos = tuple(np.argwhere(maze == START)[0])
            goal_pos = tuple(np.argwhere(maze == GOAL)[0])
            if check_maze_solvability(maze, start_pos, goal_pos):
                return maze

    def _init_render(self):
        pygame.display.set_caption("Pixel Paladin RL - Enhanced Maze Environment")
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        self.background_img = pygame.Surface((self.screen_width, self.screen_height))
        color1 = (30, 30, 50)
        color2 = (10, 10, 20)
        for y in range(self.screen_height):
            r = color1[0] + (color2[0] - color1[0]) * y / self.screen_height
            g = color1[1] + (color2[1] - color1[1]) * y / self.screen_height
            b = color1[2] + (color2[2] - color1[2]) * y / self.screen_height
            pygame.draw.line(self.background_img, (r, g, b), (0, y), (self.screen_width, y))

    def _load_sprites(self):
        block_size = self.grid_size
        
        # Wall sprite
        wall = pygame.Surface((block_size, block_size))
        wall.fill(GRAY)
        for _ in range(10):
            x, y = random.randint(0, block_size-1), random.randint(0, block_size-1)
            radius = random.randint(1, 3)
            color_var = random.randint(-20, 20)
            color = max(0, min(255, GRAY[0] + color_var))
            pygame.draw.circle(wall, (color, color, color), (x, y), radius)
        pygame.draw.rect(wall, (50, 50, 50), wall.get_rect(), 1)
        self.sprites['wall'] = wall
        
        # Floor sprite
        floor = pygame.Surface((block_size, block_size))
        floor.fill(LIGHT_BLUE)
        for _ in range(5):
            x, y = random.randint(0, block_size-1), random.randint(0, block_size-1)
            radius = random.randint(1, 2)
            pygame.draw.circle(floor, (200, 220, 255), (x, y), radius)
        pygame.draw.rect(floor, (130, 190, 220), floor.get_rect(), 1)
        self.sprites['floor'] = floor
        
        # Start sprite
        start = pygame.Surface((block_size, block_size))
        start.fill(BLUE)
        pygame.draw.rect(start, (0, 0, 200), start.get_rect(), 2)
        self.sprites['start'] = start
        
        # Goal sprite - made more visible
        goal = pygame.Surface((block_size, block_size))
        goal.fill(GREEN)
        # Add a star pattern to make it more distinctive
        pygame.draw.polygon(goal, (255, 255, 0), [
            (block_size//2, block_size//4),
            (block_size//3, block_size*3//4),
            (block_size*3//4, block_size//3),
            (block_size//4, block_size//3),
            (block_size*2//3, block_size*3//4)
        ])
        pygame.draw.rect(goal, (0, 200, 0), goal.get_rect(), 2)
        self.sprites['goal'] = goal
        
        # Agent sprite
        agent = pygame.Surface((block_size, block_size), pygame.SRCALPHA)
        agent_color = RED
        pygame.draw.circle(agent, agent_color, (block_size//2, block_size//2), block_size//2-2)
        pygame.draw.circle(agent, WHITE, (block_size//2-3, block_size//2-2), 2)
        pygame.draw.circle(agent, WHITE, (block_size//2+3, block_size//2-2), 2)
        self.sprites['agent'] = agent
        
        # Player sprite
        player = pygame.Surface((block_size, block_size), pygame.SRCALPHA)
        player_color = PURPLE
        pygame.draw.circle(player, player_color, (block_size//2, block_size//2), block_size//2-2)
        pygame.draw.circle(player, WHITE, (block_size//2-3, block_size//2-2), 2)
        pygame.draw.circle(player, WHITE, (block_size//2+3, block_size//2-2), 2)
        self.sprites['player'] = player
        
        # Path sprite
        path = pygame.Surface((block_size, block_size), pygame.SRCALPHA)
        path_color = (255, 255, 0, 100)
        pygame.draw.circle(path, path_color, (block_size//2, block_size//2), block_size//4)
        self.sprites['path'] = path

    def _create_ui_elements(self):
        button_width = 120
        button_height = 30
        margin = 10
        
        self.buttons = {
            'new_maze': Button(margin, SCREEN_HEIGHT - 3*button_height - 3*margin, 
                              button_width, button_height, "New Maze"),
            'toggle_mode': Button(margin, SCREEN_HEIGHT - 2*button_height - 2*margin,
                                 button_width, button_height, "Mode: AI"),
            'race': Button(margin, SCREEN_HEIGHT - button_height - margin,
                          button_width, button_height, "Start Race")
        }

    def _generate_maze(self):
        if self.generation_method == 'dfs':
            return self._generate_dfs_maze()
        elif self.generation_method == 'prim':
            return self._generate_prims_maze()
        else:
            return self._generate_default_maze()

    def _generate_dfs_maze(self):
        maze = np.ones((self.height, self.width), dtype=int) * WALL
        
        stack = []
        start_r, start_c = 1, 1
        stack.append((start_r, start_c))
        maze[start_r, start_c] = EMPTY
        
        directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        
        while stack:
            current_r, current_c = stack[-1]
            
            neighbors = []
            for dr, dc in directions:
                nr, nc = current_r + dr, current_c + dc
                if (0 <= nr < self.height and 0 <= nc < self.width and maze[nr, nc] == WALL):
                    neighbors.append((nr, nc, dr//2, dc//2))
            
            if neighbors:
                next_r, next_c, dr, dc = random.choice(neighbors)
                maze[current_r + dr, current_c + dc] = EMPTY
                maze[next_r, next_c] = EMPTY
                stack.append((next_r, next_c))
            else:
                stack.pop()
        
        start_r, start_c = 1, 1
        goal_r, goal_c = self.height - 2, self.width - 2
        
        maze[start_r, start_c] = START
        maze[goal_r, goal_c] = GOAL
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = start_r + dr, start_c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                if maze[nr, nc] == WALL and random.random() < 0.7:
                    maze[nr, nc] = EMPTY
                    
            nr, nc = goal_r + dr, goal_c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                if maze[nr, nc] == WALL and random.random() < 0.7:
                    maze[nr, nc] = EMPTY
        
        return maze

    def _generate_prims_maze(self):
        maze = np.ones((self.height, self.width), dtype=int) * WALL
        
        start_r, start_c = 1, 1
        maze[start_r, start_c] = EMPTY
        
        walls = []
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = start_r + dr, start_c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                walls.append((nr, nc))
        
        while walls:
            wall_idx = random.randint(0, len(walls) - 1)
            wall_r, wall_c = walls.pop(wall_idx)
            
            cells_in_maze = 0
            cells_not_in_maze = []
            
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = wall_r + dr, wall_c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    if maze[nr, nc] == EMPTY:
                        cells_in_maze += 1
                    else:
                        cells_not_in_maze.append((nr, nc))
            
            if cells_in_maze == 1:
                maze[wall_r, wall_c] = EMPTY
                
                for nr, nc in cells_not_in_maze:
                    if maze[nr, nc] == WALL:
                        maze[nr, nc] = EMPTY
                        
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nnr, nnc = nr + dr, nc + dc
                            if 0 <= nnr < self.height and 0 <= nnc < self.width and maze[nnr, nnc] == WALL:
                                walls.append((nnr, nnc))
        
        start_r, start_c = 1, 1
        goal_r, goal_c = self.height - 2, self.width - 2
        
        maze[start_r, start_c] = START
        maze[goal_r, goal_c] = GOAL
        
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = start_r + dr, start_c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width and maze[nr, nc] == WALL:
                maze[nr, nc] = EMPTY
                
            nr, nc = goal_r + dr, goal_c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width and maze[nr, nc] == WALL:
                maze[nr, nc] = EMPTY
        
        return maze

    def _generate_default_maze(self):
        maze = np.ones((self.height, self.width), dtype=int) * WALL
        
        maze[1:self.height-1, 1:self.width-1] = EMPTY
        
        for r in range(2, self.height - 1, 2):
            for c in range(1, self.width - 1):
                if random.random() < 0.7:
                    maze[r, c] = WALL
            
            passages = random.sample(range(1, self.width - 1), max(1, self.width // 5))
            for c in passages:
                maze[r, c] = EMPTY
        
        for c in range(2, self.width - 1, 2):
            for r in range(1, self.height - 1):
                if random.random() < 0.7:
                    maze[r, c] = WALL
            
            passages = random.sample(range(1, self.height - 1), max(1, self.height // 5))
            for r in passages:
                maze[r, c] = EMPTY
        
        maze[1, 1] = START
        maze[self.height - 2, self.width - 2] = GOAL
        
        for dr, dc in [(0, 1), (1, 0)]:
            maze[1 + dr, 1 + dc] = EMPTY
            maze[self.height - 2 - dr, self.width - 2 - dc] = EMPTY
        
        return maze

    def _validate_maze(self):
        if not isinstance(self.maze, np.ndarray) or self.maze.ndim != 2:
            return False
        if self.maze.shape != (self.height, self.width):
            return False
        if np.count_nonzero(self.maze == START) != 1:
            return False
        if np.count_nonzero(self.maze == GOAL) != 1:
            return False
        return True

    def _find_pos(self, element_type):
        pos = np.argwhere(self.maze == element_type)
        return tuple(pos[0]) if len(pos) > 0 else None

    def reset(self):
        # Only generate new maze if in dynamic mode or when explicitly requested
        if self.maze_type == 'dynamic':
            self.maze = self._generate_solvable_maze()
            self.start_pos = self._find_pos(START)
            self.goal_pos = self._find_pos(GOAL)
        
        self.agent_pos = self.start_pos
        self.current_step = 0
        self.ai_finish_time = 0
        
        # Don't reset player position in race mode
        if self.game_mode == 'player':
            self.player_pos = self.start_pos
            self.player_moves = 0
            self.player_finish_time = 0
        elif self.game_mode == 'race' and not self.race_started:
            self.player_pos = self.start_pos
            self.player_moves = 0
            self.player_finish_time = 0
        
        self.ai_path = []
        self.show_path = False
        
        return self._get_state()

    def step(self, action):
        if action not in self.action_space:
            raise ValueError(f"Invalid action: {action}")

        prev_pos = self.agent_pos
        row, col = self.agent_pos

        if action == 0:  # Up
            row -= 1
        elif action == 1:  # Down
            row += 1
        elif action == 2:  # Left
            col -= 1
        elif action == 3:  # Right
            col += 1

        if 0 <= row < self.height and 0 <= col < self.width and self.maze[row, col] != WALL:
            self.agent_pos = (row, col)

        reward = -0.1  # Small penalty for each step to encourage efficiency
        terminated = False
        truncated = False

        if self.agent_pos == self.goal_pos:
            reward = 10.0  # Large reward for reaching the goal
            terminated = True
            self.ai_finish_time = self.current_step
        elif self.agent_pos == prev_pos:  # Hit a wall
            reward = -1.0
        elif self.current_step >= self.max_steps:
            truncated = True
            reward = -5.0  # Penalty for not reaching goal

        self.current_step += 1

        done = terminated or truncated
        state = self._get_state()
        info = {}

        return state, reward, terminated, truncated, info

    def player_step(self, action):
        if action not in self.action_space:
            return False
        
        prev_pos = self.player_pos
        row, col = self.player_pos
        
        if action == 0:
            row -= 1
        elif action == 1:
            row += 1
        elif action == 2:
            col -= 1
        elif action == 3:
            col += 1
        
        if 0 <= row < self.height and 0 <= col < self.width and self.maze[row, col] != WALL:
            self.player_pos = (row, col)
            self.player_moves += 1
            
            if self.player_pos == self.goal_pos:
                self.player_finish_time = self.player_moves
            
            return True
        
        return False

    def _get_state(self):
        return self.agent_pos

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()
            
            for button_name, button in self.buttons.items():
                if button.is_clicked(event):
                    if button_name == 'new_maze':
                        self.maze = self._generate_solvable_maze()
                        self.start_pos = self._find_pos(START)
                        self.goal_pos = self._find_pos(GOAL)
                        self.reset()
                    elif button_name == 'toggle_mode':
                        if self.game_mode == 'ai':
                            self.game_mode = 'player'
                            self.buttons['toggle_mode'].text = "Mode: Player"
                            self.player_pos = self.start_pos
                        elif self.game_mode == 'player':
                            self.game_mode = 'ai'
                            self.buttons['toggle_mode'].text = "Mode: AI"
                        self.reset()
                    elif button_name == 'race':
                        self.game_mode = 'race'
                        self.buttons['toggle_mode'].text = "Mode: Race"
                        self.reset()
                        self.race_started = True
            
            if event.type == pygame.KEYDOWN and (self.game_mode == 'player' or self.game_mode == 'race'):
                if event.key == pygame.K_UP:
                    self.player_step(0)
                elif event.key == pygame.K_DOWN:
                    self.player_step(1)
                elif event.key == pygame.K_LEFT:
                    self.player_step(2)
                elif event.key == pygame.K_RIGHT:
                    self.player_step(3)
        
        mouse_pos = pygame.mouse.get_pos()
        for button in self.buttons.values():
            button.check_hover(mouse_pos)

    def _draw_ui(self):
        ui_panel = pygame.Rect(0, SCREEN_HEIGHT - 100, SCREEN_WIDTH, 100)
        pygame.draw.rect(self.screen, UI_BG, ui_panel)
        
        for button in self.buttons.values():
            button.draw(self.screen, self.ui_font)
        
        info_x = 150
        info_y = SCREEN_HEIGHT - 90
        
        title_surf = self.ui_title_font.render("Pixel Paladin Maze Runner", True, UI_TEXT)
        self.screen.blit(title_surf, (info_x, info_y))
        
        mode_text = f"Mode: {self.game_mode.upper()}"
        steps_text = f"AI Steps: {self.current_step}" if self.game_mode != 'player' else f"Player Moves: {self.player_moves}"
        
        mode_surf = self.ui_font.render(mode_text, True, UI_TEXT)
        steps_surf = self.ui_font.render(steps_text, True, UI_TEXT)
        
        self.screen.blit(mode_surf, (info_x, info_y + 30))
        self.screen.blit(steps_surf, (info_x, info_y + 50))
        
        if self.game_mode == 'player' or self.game_mode == 'race':
            controls_text = "Controls: Arrow Keys to move"
            controls_surf = self.ui_font.render(controls_text, True, UI_TEXT)
            self.screen.blit(controls_surf, (info_x + 300, info_y + 30))

        if self.game_mode == 'race':
            steps_text = f"AI Steps: {self.current_step} | Player Moves: {self.player_moves}"
            steps_surf = self.ui_font.render(steps_text, True, UI_TEXT)
            self.screen.blit(steps_surf, (info_x, info_y + 50))
            
            if self.player_finish_time > 0 and self.ai_finish_time > 0:
                if self.player_finish_time < self.ai_finish_time:
                    result_text = f"Player Won! ({self.player_finish_time} vs {self.ai_finish_time})"
                else:
                    result_text = f"AI Won! ({self.ai_finish_time} vs {self.player_finish_time})"
                result_surf = self.ui_font.render(result_text, True, UI_HIGHLIGHT)
                self.screen.blit(result_surf, (info_x, info_y + 70))
            elif self.player_finish_time > 0:
                result_text = f"Player Finished in {self.player_finish_time} moves!"
                result_surf = self.ui_font.render(result_text, True, UI_HIGHLIGHT)
                self.screen.blit(result_surf, (info_x, info_y + 70))
            elif self.ai_finish_time > 0:
                result_text = f"AI Finished in {self.ai_finish_time} steps!"
                result_surf = self.ui_font.render(result_text, True, UI_HIGHLIGHT)
                self.screen.blit(result_surf, (info_x, info_y + 70))

    def _draw_maze(self):
        self.screen.blit(self.background_img, (0, 0))
        
        for r in range(self.height):
            for c in range(self.width):
                rect = pygame.Rect(c * self.grid_size, r * self.grid_size, 
                                 self.grid_size, self.grid_size)
                
                if self.maze[r, c] == WALL:
                    self.screen.blit(self.sprites['wall'], rect)
                elif self.maze[r, c] == START:
                    self.screen.blit(self.sprites['start'], rect)
                elif self.maze[r, c] == GOAL:
                    self.screen.blit(self.sprites['goal'], rect)
                else:
                    self.screen.blit(self.sprites['floor'], rect)
        
        if self.show_path and self.ai_path:
            for pos in self.ai_path:
                r, c = pos
                rect = pygame.Rect(c * self.grid_size, r * self.grid_size,
                                 self.grid_size, self.grid_size)
                self.screen.blit(self.sprites['path'], rect)
        
        if self.game_mode != 'player':
            r, c = self.agent_pos
            rect = pygame.Rect(c * self.grid_size, r * self.grid_size,
                             self.grid_size, self.grid_size)
            self.screen.blit(self.sprites['agent'], rect)
        
        if self.game_mode == 'player' or self.game_mode == 'race':
            r, c = self.player_pos
            rect = pygame.Rect(c * self.grid_size, r * self.grid_size,
                             self.grid_size, self.grid_size)
            self.screen.blit(self.sprites['player'], rect)

    def render(self):
        if self.render_mode != 'human':
            return
        
        self._handle_events()
        self._draw_maze()
        self._draw_ui()
        
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    def get_shortest_path(self):
        return find_shortest_path(self.maze, self.start_pos, self.goal_pos)

    def toggle_path_display(self):
        self.show_path = not self.show_path
        if self.show_path:
            self.ai_path = self.get_shortest_path()

# --- Q-Learning Agent ---
class QLearningAgent:
    def __init__(self, env, learning_rate=DEFAULT_LEARNING_RATE, 
                 discount_factor=DEFAULT_DISCOUNT_FACTOR, 
                 epsilon=DEFAULT_EPSILON, 
                 epsilon_decay=DEFAULT_EPSILON_DECAY,
                 min_epsilon=DEFAULT_MIN_EPSILON):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        self.q_table = np.zeros((env.height, env.width, env.n_actions))
        
        self.rewards = []
        self.steps_per_episode = []
        
        self.best_path = None
        self.best_reward = -np.inf

    def get_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.choice(self.env.action_space)
        
        row, col = state
        return np.argmax(self.q_table[row, col])

    def update_q_table(self, state, action, reward, next_state, done):
        row, col = state
        next_row, next_col = next_state
        
        current_q = self.q_table[row, col, action]
        
        if done:
            target = reward
        else:
            max_next_q = np.max(self.q_table[next_row, next_col])
            target = reward + self.discount_factor * max_next_q
        
        self.q_table[row, col, action] += self.learning_rate * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def train(self, episodes=DEFAULT_EPISODES, render_every=DEFAULT_RENDER_EVERY):
        for episode in range(1, episodes + 1):
            state = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                self.update_q_table(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
                
                if render_every and episode % render_every == 0:
                    self.env.render()
                    time.sleep(0.01)
            
            self.rewards.append(total_reward)
            self.steps_per_episode.append(steps)
            self.decay_epsilon()
            
            if total_reward > self.best_reward:
                self.best_reward = total_reward
                self.best_path = self._extract_path()
            
            if episode % 100 == 0:
                print(f"Episode: {episode}, Reward: {total_reward:.1f}, Steps: {steps}, Epsilon: {self.epsilon:.3f}")

    def _extract_path(self):
        path = []
        state = self.env.start_pos
        visited = set()
        
        while state != self.env.goal_pos:
            if state in visited:
                break
            visited.add(state)
            path.append(state)
            
            action = np.argmax(self.q_table[state[0], state[1]])
            
            row, col = state
            if action == 0:
                row -= 1
            elif action == 1:
                row += 1
            elif action == 2:
                col -= 1
            elif action == 3:
                col += 1
            
            if (0 <= row < self.env.height and 0 <= col < self.env.width and 
                self.env.maze[row, col] != WALL):
                state = (row, col)
            else:
                break
        
        if state == self.env.goal_pos:
            path.append(state)
        
        return path

    def plot_learning_curve(self):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.rewards)
        plt.title('Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.steps_per_episode)
        plt.title('Steps per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        
        plt.tight_layout()
        plt.show()

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description='Maze RL Environment')
    parser.add_argument('--episodes', type=int, default=DEFAULT_EPISODES,
                       help='Number of training episodes')
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE,
                       help='Learning rate for Q-learning')
    parser.add_argument('--discount_factor', type=float, default=DEFAULT_DISCOUNT_FACTOR,
                       help='Discount factor for future rewards')
    parser.add_argument('--epsilon', type=float, default=DEFAULT_EPSILON,
                       help='Initial exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=DEFAULT_EPSILON_DECAY,
                       help='Decay rate for exploration')
    parser.add_argument('--min_epsilon', type=float, default=DEFAULT_MIN_EPSILON,
                       help='Minimum exploration rate')
    parser.add_argument('--render_every', type=int, default=DEFAULT_RENDER_EVERY,
                       help='Render every N episodes during training')
    parser.add_argument('--maze_type', choices=['static', 'dynamic'], default='static',
                       help='Type of maze (static or dynamic)')
    parser.add_argument('--generation_method', choices=['dfs', 'prim', 'default'], default='dfs',
                       help='Maze generation method')
    parser.add_argument('--no_train', action='store_true',
                       help='Skip training and go straight to visualization')
    args = parser.parse_args()

    env = MazeEnv(render_mode='human', maze_type=args.maze_type, 
                 generation_method=args.generation_method)
    
    agent = QLearningAgent(env, 
                          learning_rate=args.learning_rate,
                          discount_factor=args.discount_factor,
                          epsilon=args.epsilon,
                          epsilon_decay=args.epsilon_decay,
                          min_epsilon=args.min_epsilon)
    
    if not args.no_train:
        print("Starting training...")
        agent.train(episodes=args.episodes, render_every=args.render_every)
        agent.plot_learning_curve()
    
    print("Entering interactive mode...")
    print("Controls:")
    print("  - Arrow keys: Move player (in player/race mode)")
    print("  - N: Generate new maze")
    print("  - M: Toggle game mode")
    print("  - P: Toggle path display")
    print("  - R: Start race")
    print("  - Q: Quit")
    
    running = True
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_n:
                    env.maze = env._generate_solvable_maze()
                    env.start_pos = env._find_pos(START)
                    env.goal_pos = env._find_pos(GOAL)
                    env.reset()
                elif event.key == pygame.K_m:
                    if env.game_mode == 'ai':
                        env.game_mode = 'player'
                        env.player_pos = env.start_pos
                    elif env.game_mode == 'player':
                        env.game_mode = 'ai'
                    env.reset()
                elif event.key == pygame.K_p:
                    env.toggle_path_display()
                elif event.key == pygame.K_r:
                    env.game_mode = 'race'
                    env.reset()
                    env.race_started = True
        
        if env.game_mode == 'race' and env.race_started:
            if not env.ai_finish_time and env.agent_pos != env.goal_pos:
                state = env._get_state()
                action = agent.get_action(state, training=False)
                env.step(action)
            
            env.render()
            clock.tick(10)
        else:
            env.render()
            clock.tick(60)
    
    env.close()

if __name__ == "__main__":
    main()