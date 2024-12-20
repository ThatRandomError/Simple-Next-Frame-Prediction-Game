import pyn
import pygame
import sys

nn = pyn.NN([pyn.Layer(101, pyn.relu, pyn.relu_derivative),
            pyn.Layer(64, pyn.relu, pyn.relu_derivative),
            pyn.Layer(32, pyn.relu, pyn.relu_derivative),
            pyn.Layer(16, pyn.relu, pyn.relu_derivative),
            pyn.Layer(100, pyn.sigmoid, pyn.sigmoid_derivative)])

nn.load("model1.pyn")



pygame.init()

pixel_size = 50

grid_width, grid_height = 10, 10
width, height = grid_width * pixel_size, grid_height * pixel_size

screen = pygame.display.set_mode((width, height))

pixel_values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

def draw():
    global pixel_values
    surface = pygame.Surface((grid_width, grid_height))
    for i in range(grid_height):
        for j in range(grid_width):
            pixel_value = pixel_values[i * grid_width + j]
            color_value = int(pixel_value * 255)
            surface.set_at((j, i), (color_value, color_value, color_value))

    scaled_surface = pygame.transform.scale(surface, (width, height))

    screen.blit(scaled_surface, (0, 0))

    pygame.display.flip()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        left = 1
    else:
        left = 0
    if keys[pygame.K_RIGHT]:
        right = 1
    else:
        right = 0
    if keys[pygame.K_SPACE]:
        jumping = 1
    else:
        jumping = 0

    pixel_values.append(left)
    pixel_values.append(right)
    pixel_values.append(jumping)

    pixel_values = nn.forward(pixel_values).tolist()

    print(left)
    print(right)
    print(jumping)

    draw()

pygame.quit()
sys.exit()
