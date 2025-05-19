import pygame
import sys
import math
from raypy import Vector, Ray, Lens, Mirror, Scene, Simulator, LightSource

# --- Config ---
WIDTH, HEIGHT = 1000, 700
BG_COLOR = (30, 30, 30)
FPS = 60

# --- Parameters ---
num_rays = 11
angle_range = (-math.radians(15), math.radians(15))
source_pos = Vector(-4, 0)  # scene space

# --- Coordinate transforms ---
def to_screen(vec, scene_size, screen_size):
    x = int((vec.x / scene_size[0] + 0.5) * screen_size[0])
    y = int((0.5 - vec.y / scene_size[1]) * screen_size[1])
    return (x, y)

def from_screen(pos, scene_size, screen_size):
    x = (pos[0] / screen_size[0] - 0.5) * scene_size[0]
    y = (0.5 - pos[1] / screen_size[1]) * scene_size[1]
    return Vector(x, y)

# --- Setup ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ray Optics Simulator (Pygame)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 16)

# --- Scene setup ---
sim = Simulator()
scene = sim.create_scene(size=(10, 7))

# Add default elements
scene.add_optical_element(Lens(Vector(0, 0), width=2, focal_length=3))
scene.add_optical_element(Mirror(Vector(2, -1), Vector(4, 1)))
scene.add_light_source(LightSource(position=source_pos, num_rays=num_rays, angle_range=angle_range))
sim.simulate()

# --- Drawing functions ---
def draw_path(path, color=(255, 255, 100)):
    if len(path) < 2:
        return
    points = [to_screen(Vector(x, y), scene.size, (WIDTH, HEIGHT)) for (x, y) in path]
    pygame.draw.lines(screen, color, False, points, 1)

def draw_scene():
    screen.fill(BG_COLOR)

    # Draw rays
    for ray in scene.rays:
        color = tuple(int(255 * c) for c in ray.get_color()[:3])
        draw_path(ray.path, color)

    # Draw lens and mirror
    for obj in scene.optical_elements:
        if isinstance(obj, Lens):
            start = to_screen(obj.start_point, scene.size, (WIDTH, HEIGHT))
            end = to_screen(obj.end_point, scene.size, (WIDTH, HEIGHT))
            pygame.draw.line(screen, (120, 120, 255), start, end, 3)
        elif isinstance(obj, Mirror):
            start = to_screen(obj.start_point, scene.size, (WIDTH, HEIGHT))
            end = to_screen(obj.end_point, scene.size, (WIDTH, HEIGHT))
            pygame.draw.line(screen, (200, 200, 200), start, end, 2)

    # Draw source
    for src in scene.light_sources:
        src_pos = to_screen(src.position, scene.size, (WIDTH, HEIGHT))
        pygame.draw.circle(screen, (255, 200, 0), src_pos, 5)

    # Display text
    info = font.render(f"Rays: {num_rays} | +/- to change | Click to move source", True, (200, 200, 200))
    screen.blit(info, (10, 10))
    pygame.display.flip()

# --- Main loop ---
dragging_source = False
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            mouse_vec = from_screen((mx, my), scene.size, (WIDTH, HEIGHT))
            for src in scene.light_sources:
                if (mouse_vec - src.position).magnitude() < 0.5:
                    dragging_source = True

        elif event.type == pygame.MOUSEBUTTONUP:
            dragging_source = False

        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                num_rays = min(101, num_rays + 2)
            elif event.key == pygame.K_MINUS:
                num_rays = max(1, num_rays - 2)
            elif event.key == pygame.K_r:
                scene.optical_elements.clear()
                scene.light_sources.clear()
                scene.add_light_source(LightSource(position=source_pos, num_rays=num_rays, angle_range=angle_range))
            elif event.key == pygame.K_l:
                mx, my = pygame.mouse.get_pos()
                pos = from_screen((mx, my), scene.size, (WIDTH, HEIGHT))
                scene.add_optical_element(Lens(pos, width=2, focal_length=3))
            elif event.key == pygame.K_m:
                mx, my = pygame.mouse.get_pos()
                pos = from_screen((mx, my), scene.size, (WIDTH, HEIGHT))
                scene.add_optical_element(Mirror(pos - Vector(1, 0), pos + Vector(1, 0)))

            # Update rays after any change
            for src in scene.light_sources:
                src.num_rays = num_rays
            sim.simulate()

    if dragging_source:
        mx, my = pygame.mouse.get_pos()
        new_pos = from_screen((mx, my), scene.size, (WIDTH, HEIGHT))
        scene.light_sources[0].position = new_pos
        sim.simulate()

    draw_scene()
    clock.tick(FPS)

pygame.quit()
sys.exit()
