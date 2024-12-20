import pygame
import sys
import pickle
import math
from PIL import Image
import copy

pygame.init()

WIDTH, HEIGHT = 10, 10

BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
WHITE = (255, 255, 255)

PLAYER_COLOR = BLACK
PLAYER_SPEED = 0.25
PLAYER_JUMP = 0.5

GRAVITY = 0.1

SCALE = 50

PLATFORM_COLOR = GRAY

screen = pygame.display.set_mode((WIDTH * SCALE, HEIGHT * SCALE))
surface = pygame.Surface((WIDTH, HEIGHT))
pygame.display.set_caption("Platformer")

clock = pygame.time.Clock()
FPS = 60

platforms = [
    pygame.Rect(2, 8, 4, 1),
    pygame.Rect(5, 6, 3, 1),
    pygame.Rect(1, 4, 2, 1),
    pygame.Rect(7, 2, 2, 1),
]

player_x, player_y = 1, 1
old_player_x, old_player_y = player_x, player_y
player_velocity_y = 0

is_jumping = False

running = True
while running:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and player_x > 0:
        player_x -= PLAYER_SPEED
        player_state = -2
    if keys[pygame.K_RIGHT] and player_x < WIDTH - 1:
        player_x += PLAYER_SPEED
        player_state = 2
    if keys[pygame.K_SPACE] and not is_jumping:
        player_state = 1
        player_velocity_y = -PLAYER_JUMP
        is_jumping = True
    else:
        player_state = 0

    if (player_velocity_y > 0):
        player_state = -1

    player_velocity_y += GRAVITY
    player_y = int(player_y)
    player_y += player_velocity_y

    for platform in platforms:
        if (
            platform.collidepoint(player_x, player_y + 1)
            and player_velocity_y > 0
        ):
            player_y = platform.top - 1
            player_velocity_y = 0
            is_jumping = False

    if player_y > HEIGHT - 1:
        player_x, player_y = 1, 1
        player_velocity_y = 0
        is_jumping = False

    # DRAWING
    surface.fill(WHITE)

    for platform in platforms:
        pygame.draw.rect(surface, PLATFORM_COLOR, platform)

    pygame.draw.rect(surface, PLAYER_COLOR, pygame.Rect(player_x, player_y, 1, 1))

    surface_scaled = pygame.Surface((WIDTH * SCALE, HEIGHT * SCALE))
    surface_scaled = pygame.transform.scale_by(surface, SCALE)
    screen.blit(surface_scaled, (0, 0))
    pygame.display.flip()

pygame.quit()
sys.exit()

