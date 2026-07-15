#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import vispy
from vispy.scene import SceneCanvas, visuals
from vispy.visuals.transforms import STTransform


# ============================================================
# Pheromone Boids parameters
# ============================================================

# Boids
N = 180
WORLD_MIN = -1.0
WORLD_MAX = 1.0
MIN_VEL = 0.006
MAX_VEL = 0.026

# Basic Boids forces
COHESION_FORCE = 0.0075
SEPARATION_FORCE = 0.020
ALIGNMENT_FORCE = 0.050
COHESION_DISTANCE = 0.25
SEPARATION_DISTANCE = 0.045
ALIGNMENT_DISTANCE = 0.18
BOUNDARY_FORCE = 0.010
WANDER_FORCE = 0.0018
WANDER_ANGLE_STD = np.pi / 7

# Pheromone field
GRID_SIZE = 180
EVAPORATION_RATE = 0.990
DIFFUSION_RATE = 0.090
GROUP_PHEROMONE_DEPOSIT = 0.020
FOOD_PHEROMONE_DEPOSIT = 0.120
TRAIL_PHEROMONE_DEPOSIT = 0.012
ALARM_PHEROMONE_DEPOSIT = 0.090
DEPOSIT_RADIUS = 2
VECTOR_DEPOSIT_DISTANCE = 7
VECTOR_DEPOSIT_ANGLE = np.pi / 6
FRAME_DT = 1.0 / 60.0
PHEROMONE_EVAPORATION_DELAY_SECONDS = 5.0
PHEROMONE_LIFETIME_FRAMES = int(PHEROMONE_EVAPORATION_DELAY_SECONDS / FRAME_DT)

# Pheromone is generated only when local interaction occurs.
GROUP_PHEROMONE_MIN_NEIGHBORS = 5
GROUP_PHEROMONE_DISTANCE = 0.08
GROUP_PHEROMONE_REQUIRED_SECONDS = 1.2
GROUP_PHEROMONE_REQUIRED_FRAMES = int(GROUP_PHEROMONE_REQUIRED_SECONDS / FRAME_DT)
GROUP_CONTACT_DECAY = 2

# Pheromone sensing
SENSOR_DISTANCE = 0.105
SENSOR_ANGLE = np.pi / 4
ATTRACT_PHEROMONE_FORCE = 0.0022
TRAIL_PHEROMONE_FORCE = 0.0035
GROUP_PHEROMONE_FORCE = 0.0025
ALARM_PHEROMONE_FORCE = 0.013
PHEROMONE_DIRECTION_SHARPNESS = 2.0

# Optional food / danger system
FOOD_POSITIONS = np.array([
    [0.72, 0.70],
    [-0.72, 0.62],
])
DANGER_POSITIONS = np.array([
    [0.18, -0.62],
])
FOOD_RESOURCE_CAPACITY = 100
FOOD_REACTION_LIMIT = FOOD_RESOURCE_CAPACITY
DANGER_SPEED = 0.004
FOOD_DETECTION_RADIUS = 0.16
FOOD_ATTRACTION_RADIUS = 0.30
DANGER_DETECTION_RADIUS = 0.18
FOOD_ATTRACTION_FORCE = 0.002
FOOD_LEAVE_FORCE = 0.018
FOOD_DEPARTURE_TRAIL_MULTIPLIER = 4.0
DANGER_AVOIDANCE_FORCE = 0.035

# Display
CANVAS_SIZE = 720
BOID_LENGTH = 4.2
BOID_WIDTH = 2.4
FOOD_SIZE = 13
DANGER_SIZE = 15


def normalize(vector):
    """Return a unit vector. A zero vector is returned unchanged."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def rotate_2d(vector, angle):
    """Rotate a 2D vector by angle radians."""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        c * vector[0] - s * vector[1],
        s * vector[0] + c * vector[1],
    ])


def limit_velocity(velocity):
    """Keep velocity between MIN_VEL and MAX_VEL."""
    speed = np.linalg.norm(velocity)
    if speed == 0:
        return np.array([MIN_VEL, 0.0])
    if speed < MIN_VEL:
        return velocity / speed * MIN_VEL
    if speed > MAX_VEL:
        return velocity / speed * MAX_VEL
    return velocity


def world_to_grid(position):
    """Convert world coordinates [-1, 1] to pheromone-grid indexes."""
    normalized = (position - WORLD_MIN) / (WORLD_MAX - WORLD_MIN)
    grid = np.floor(normalized * (GRID_SIZE - 1)).astype(int)
    return np.clip(grid, 0, GRID_SIZE - 1)


def grid_to_display(position):
    """Convert world coordinates to visualizer pixel coordinates."""
    normalized = (position - WORLD_MIN) / (WORLD_MAX - WORLD_MIN)
    return normalized * (GRID_SIZE - 1)


def grid_to_display_3d(position, z_value):
    """Convert world coordinates to visualizer coordinates with a z layer."""
    xy = grid_to_display(position)
    z = np.full((xy.shape[0], 1), z_value)
    return np.column_stack((xy, z))


def sample_pheromone(field, position):
    """Read pheromone concentration at a world coordinate."""
    gx, gy = world_to_grid(position)
    return field[gy, gx]


def create_pheromone_layer():
    """Create amount, direction, and age fields for one pheromone type."""
    return {
        'amount': np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32),
        'direction_x': np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32),
        'direction_y': np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32),
        'age': np.full((GRID_SIZE, GRID_SIZE), PHEROMONE_LIFETIME_FRAMES, dtype=np.float32),
    }


def sample_vector_pheromone(layer, position):
    """Read amount and direction from a vector pheromone layer."""
    gx, gy = world_to_grid(position)
    amount = layer['amount'][gy, gx]
    age = layer['age'][gy, gx]
    direction = np.array([layer['direction_x'][gy, gx], layer['direction_y'][gy, gx]])
    return amount, normalize(direction), age


def deposit_pheromone(field, position, amount):
    """Deposit pheromone around a position using a small radial kernel."""
    gx, gy = world_to_grid(position)
    for dy in range(-DEPOSIT_RADIUS, DEPOSIT_RADIUS + 1):
        for dx in range(-DEPOSIT_RADIUS, DEPOSIT_RADIUS + 1):
            distance = np.sqrt(dx * dx + dy * dy)
            if distance > DEPOSIT_RADIUS:
                continue
            x_index = gx + dx
            y_index = gy + dy
            if 0 <= x_index < GRID_SIZE and 0 <= y_index < GRID_SIZE:
                field[y_index, x_index] += amount * (1.0 - distance / (DEPOSIT_RADIUS + 1))


def deposit_vector_pheromone(layer, position, velocity, amount):
    """Deposit pheromone in a forward cone and store the movement direction."""
    direction = normalize(velocity)
    if np.linalg.norm(direction) == 0:
        return

    gx, gy = world_to_grid(position)
    for dy in range(-VECTOR_DEPOSIT_DISTANCE, VECTOR_DEPOSIT_DISTANCE + 1):
        for dx in range(-VECTOR_DEPOSIT_DISTANCE, VECTOR_DEPOSIT_DISTANCE + 1):
            offset = np.array([dx, dy], dtype=float)
            distance = np.linalg.norm(offset)
            if distance == 0:
                alignment = 1.0
            elif distance <= VECTOR_DEPOSIT_DISTANCE:
                alignment = np.dot(normalize(offset), direction)
            else:
                continue

            if alignment < np.cos(VECTOR_DEPOSIT_ANGLE):
                continue

            x_index = gx + dx
            y_index = gy + dy
            if not (0 <= x_index < GRID_SIZE and 0 <= y_index < GRID_SIZE):
                continue

            weight = amount * (1.0 - distance / (VECTOR_DEPOSIT_DISTANCE + 1)) * max(alignment, 0.0)
            old_amount = layer['amount'][y_index, x_index]
            new_amount = old_amount + weight
            if new_amount <= 0:
                continue

            layer['direction_x'][y_index, x_index] = (
                layer['direction_x'][y_index, x_index] * old_amount + direction[0] * weight
            ) / new_amount
            layer['direction_y'][y_index, x_index] = (
                layer['direction_y'][y_index, x_index] * old_amount + direction[1] * weight
            ) / new_amount
            layer['amount'][y_index, x_index] = new_amount
            layer['age'][y_index, x_index] = 0.0


def refresh_pheromone_age(age_field, position):
    """Mark freshly deposited pheromone so it remains visible before evaporating."""
    gx, gy = world_to_grid(position)
    for dy in range(-DEPOSIT_RADIUS, DEPOSIT_RADIUS + 1):
        for dx in range(-DEPOSIT_RADIUS, DEPOSIT_RADIUS + 1):
            if np.sqrt(dx * dx + dy * dy) > DEPOSIT_RADIUS:
                continue
            x_index = gx + dx
            y_index = gy + dy
            if 0 <= x_index < GRID_SIZE and 0 <= y_index < GRID_SIZE:
                age_field[y_index, x_index] = 0.0


def diffuse_and_age(field, age_field):
    """Diffuse pheromone, then evaporate cells older than the configured delay."""
    neighbors = (
        np.roll(field, 1, axis=0) +
        np.roll(field, -1, axis=0) +
        np.roll(field, 1, axis=1) +
        np.roll(field, -1, axis=1)
    ) * 0.25
    age_neighbors = (
        np.roll(age_field, 1, axis=0) +
        np.roll(age_field, -1, axis=0) +
        np.roll(age_field, 1, axis=1) +
        np.roll(age_field, -1, axis=1)
    ) * 0.25

    field[:] = (field * (1.0 - DIFFUSION_RATE) + neighbors * DIFFUSION_RATE)
    age_field[:] = (age_field * (1.0 - DIFFUSION_RATE) + age_neighbors * DIFFUSION_RATE)
    age_field[:] += 1.0

    old_pheromone = age_field >= PHEROMONE_LIFETIME_FRAMES
    field[old_pheromone] *= EVAPORATION_RATE

    field[[0, -1], :] = 0.0
    field[:, [0, -1]] = 0.0
    age_field[[0, -1], :] = PHEROMONE_LIFETIME_FRAMES
    age_field[:, [0, -1]] = PHEROMONE_LIFETIME_FRAMES
    field[field < 0.0001] = 0.0


def diffuse_vector_layer(layer):
    """Diffuse amount, direction, and age for one vector pheromone layer."""
    diffuse_and_age(layer['amount'], layer['age'])
    for key in ('direction_x', 'direction_y'):
        neighbors = (
            np.roll(layer[key], 1, axis=0) +
            np.roll(layer[key], -1, axis=0) +
            np.roll(layer[key], 1, axis=1) +
            np.roll(layer[key], -1, axis=1)
        ) * 0.25
        layer[key][:] = layer[key] * (1.0 - DIFFUSION_RATE) + neighbors * DIFFUSION_RATE

    direction_norm = np.sqrt(layer['direction_x'] ** 2 + layer['direction_y'] ** 2)
    has_direction = direction_norm > 1e-6
    layer['direction_x'][has_direction] /= direction_norm[has_direction]
    layer['direction_y'][has_direction] /= direction_norm[has_direction]
    layer['direction_x'][layer['amount'] <= 0] = 0.0
    layer['direction_y'][layer['amount'] <= 0] = 0.0


def local_neighbor_counts(positions):
    """Count nearby Boids; 2 neighbors means 3 Boids including itself."""
    counts = np.zeros(len(positions), dtype=int)
    for i in range(len(positions)):
        distance = np.linalg.norm(positions - positions[i], axis=1)
        counts[i] = np.count_nonzero((distance < GROUP_PHEROMONE_DISTANCE) & (distance > 0))
    return counts


def boid_triangle_mesh(positions, velocities):
    """Build oriented triangle vertices and faces for Boid visualization."""
    centers = grid_to_display(positions)
    boid_count = len(positions)
    vertices = np.zeros((boid_count * 3, 3), dtype=np.float32)
    faces = np.arange(boid_count * 3, dtype=np.uint32).reshape(boid_count, 3)

    for i in range(boid_count):
        direction = normalize(velocities[i])
        if np.linalg.norm(direction) == 0:
            direction = np.array([1.0, 0.0])
        side = np.array([-direction[1], direction[0]])

        tip = centers[i] + direction * BOID_LENGTH
        left = centers[i] - direction * BOID_LENGTH * 0.65 + side * BOID_WIDTH
        right = centers[i] - direction * BOID_LENGTH * 0.65 - side * BOID_WIDTH

        vertices[i * 3 + 0] = (tip[0], tip[1], 30)
        vertices[i * 3 + 1] = (left[0], left[1], 30)
        vertices[i * 3 + 2] = (right[0], right[1], 30)

    return vertices, faces


def random_world_positions(count):
    """Create random points away from the world edge."""
    return np.random.uniform(-0.82, 0.82, size=(count, 2))


def initialize_danger_velocities(danger_positions):
    """Create slow constant velocities for red danger individuals."""
    angles = np.random.rand(len(danger_positions)) * 2 * np.pi
    return np.column_stack((np.cos(angles), np.sin(angles))) * DANGER_SPEED


def move_danger_positions(danger_positions, danger_velocities):
    """Move red danger individuals slowly and bounce them at the border."""
    danger_positions += danger_velocities
    for i in range(len(danger_positions)):
        for axis in range(2):
            if danger_positions[i, axis] < -0.88:
                danger_positions[i, axis] = -0.88
                danger_velocities[i, axis] *= -1
            elif danger_positions[i, axis] > 0.88:
                danger_positions[i, axis] = 0.88
                danger_velocities[i, axis] *= -1


def relocate_food(food_positions, food_reactions, food_index):
    """Move consumed food to a new point and reset its reaction count."""
    food_positions[food_index] = random_world_positions(1)[0]
    food_reactions[food_index] = 0


def food_departure_direction(position, velocity, food_position):
    """Return the direction used after a Boid reacts to food."""
    away_from_food = normalize(position - food_position)
    if np.linalg.norm(away_from_food) == 0:
        away_from_food = -normalize(velocity)
    return normalize(away_from_food + 0.35 * normalize(velocity))


def boids_force(index, positions, velocities):
    """Compute ordinary cohesion, separation, and alignment forces."""
    x_this = positions[index]
    v_this = velocities[index]
    x_that = np.delete(positions, index, axis=0)
    v_that = np.delete(velocities, index, axis=0)
    distance = np.linalg.norm(x_that - x_this, axis=1)

    coh_agents_x = x_that[distance < COHESION_DISTANCE]
    sep_agents_x = x_that[distance < SEPARATION_DISTANCE]
    ali_agents_v = v_that[distance < ALIGNMENT_DISTANCE]

    force = np.zeros(2)
    if len(coh_agents_x) > 0:
        force += COHESION_FORCE * (np.average(coh_agents_x, axis=0) - x_this)
    if len(sep_agents_x) > 0:
        force += SEPARATION_FORCE * np.sum(x_this - sep_agents_x, axis=0)
    if len(ali_agents_v) > 0:
        force += ALIGNMENT_FORCE * (np.average(ali_agents_v, axis=0) - v_this)
    return force


def pheromone_force(position, velocity, food_layer, group_layer, trail_layer, danger_layer):
    """Steer using pheromone amount, pheromone direction, and freshness."""
    direction = normalize(velocity)
    if np.linalg.norm(direction) == 0:
        direction = np.array([1.0, 0.0])

    sensor_dirs = [
        rotate_2d(direction, SENSOR_ANGLE),
        direction,
        rotate_2d(direction, -SENSOR_ANGLE),
    ]

    force = np.zeros(2)
    for sensor_dir in sensor_dirs:
        sensor_pos = position + sensor_dir * SENSOR_DISTANCE
        for layer, strength in (
                (food_layer, ATTRACT_PHEROMONE_FORCE),
                (group_layer, GROUP_PHEROMONE_FORCE),
                (trail_layer, TRAIL_PHEROMONE_FORCE)):
            amount, pheromone_direction, age = sample_vector_pheromone(layer, sensor_pos)
            if amount <= 0:
                continue
            direction_similarity = max(0.0, np.dot(direction, pheromone_direction))
            if direction_similarity <= 0:
                continue
            freshness = max(0.0, 1.0 - age / PHEROMONE_LIFETIME_FRAMES)
            response = amount * (direction_similarity ** PHEROMONE_DIRECTION_SHARPNESS) * (0.3 + 0.7 * freshness)
            force += strength * response * normalize(sensor_dir + pheromone_direction)

        amount, danger_direction, age = sample_vector_pheromone(danger_layer, sensor_pos)
        if amount > 0:
            direction_similarity = max(0.0, np.dot(direction, danger_direction))
            freshness = max(0.0, 1.0 - age / PHEROMONE_LIFETIME_FRAMES)
            response = amount * (0.3 + 0.7 * direction_similarity) * (0.3 + 0.7 * freshness)
            force -= ALARM_PHEROMONE_FORCE * response * sensor_dir

    return force


def wander_force(velocity):
    """Add a small local random steering force for autonomous exploration."""
    direction = normalize(velocity)
    if np.linalg.norm(direction) == 0:
        direction = np.array([1.0, 0.0])
    wander_direction = rotate_2d(direction, np.random.normal(0.0, WANDER_ANGLE_STD))
    return WANDER_FORCE * wander_direction


def food_and_danger_force(position, food_positions, danger_positions):
    """React directly to visible food and danger sources."""
    force = np.zeros(2)

    for food_pos in food_positions:
        offset = food_pos - position
        distance = np.linalg.norm(offset)
        if distance < FOOD_DETECTION_RADIUS:
            force -= FOOD_LEAVE_FORCE * normalize(offset)
        elif distance < FOOD_ATTRACTION_RADIUS:
            force += FOOD_ATTRACTION_FORCE * normalize(offset)

    for danger_pos in danger_positions:
        offset = position - danger_pos
        distance = np.linalg.norm(offset)
        if distance < DANGER_DETECTION_RADIUS:
            force += DANGER_AVOIDANCE_FORCE * normalize(offset)

    return force


def boundary_force(position):
    """Push Boids back inside the visible square world."""
    force = np.zeros(2)
    margin = 0.88
    if position[0] < -margin:
        force[0] += BOUNDARY_FORCE
    elif position[0] > margin:
        force[0] -= BOUNDARY_FORCE
    if position[1] < -margin:
        force[1] += BOUNDARY_FORCE
    elif position[1] > margin:
        force[1] -= BOUNDARY_FORCE
    return force


def pheromone_image(food_layer, group_layer, trail_layer, danger_layer):
    """Create a yellow trail/group/food image and a red danger image."""
    food = food_layer['amount'] / (np.max(food_layer['amount']) + 1e-6)
    group = group_layer['amount'] / (np.max(group_layer['amount']) + 1e-6)
    trail = trail_layer['amount'] / (np.max(trail_layer['amount']) + 1e-6)
    danger = danger_layer['amount'] / (np.max(danger_layer['amount']) + 1e-6)
    attract = np.clip(food * 1.00 + group * 0.65 + trail * 0.45, 0, 1)

    image = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.float32)
    image[..., 0] = np.clip(attract * 1.00 + danger * 1.00, 0, 1)
    image[..., 1] = np.clip(attract * 0.86 + danger * 0.10, 0, 1)
    image[..., 2] = np.clip(attract * 0.20 + danger * 0.08, 0, 1)
    return image


class PheromoneBoidsVisualizer(object):
    """Draw pheromone fields, Boids, food, and danger on one canvas."""

    def __init__(self):
        self._canvas = SceneCanvas(
            size=(CANVAS_SIZE, CANVAS_SIZE),
            keys='interactive',
            show=True,
            title='Pheromone Boids'
        )
        self._view = self._canvas.central_widget.add_view()
        self._view.camera = 'panzoom'
        self._view.camera.set_range(x=(0, GRID_SIZE - 1), y=(0, GRID_SIZE - 1))

        self._image = visuals.Image(np.zeros((GRID_SIZE, GRID_SIZE, 3)), parent=self._view.scene)
        self._image.transform = STTransform(translate=(0, 0, -10))
        self._image.set_gl_state('translucent', depth_test=False)

        self._boids = visuals.Mesh(parent=self._view.scene)
        self._food = visuals.Markers(parent=self._view.scene)
        self._danger = visuals.Markers(parent=self._view.scene)
        self._boids.set_gl_state('translucent', depth_test=False)
        self._food.set_gl_state('translucent', depth_test=False)
        self._danger.set_gl_state('translucent', depth_test=False)

    def update(self, positions, velocities, food_positions, danger_positions,
               food_layer, group_layer, trail_layer, danger_layer):
        self._image.set_data(pheromone_image(food_layer, group_layer, trail_layer, danger_layer))
        self._food.set_data(
            grid_to_display_3d(food_positions, 20),
            face_color=(0.1, 1.0, 0.2, 1.0),
            edge_color=(1.0, 1.0, 1.0, 1.0),
            size=FOOD_SIZE
        )
        self._danger.set_data(
            grid_to_display_3d(danger_positions, 20),
            face_color=(1.0, 0.05, 0.05, 1.0),
            edge_color=(1.0, 0.9, 0.9, 1.0),
            size=DANGER_SIZE
        )
        vertices, faces = boid_triangle_mesh(positions, velocities)
        self._boids.set_data(
            vertices=vertices,
            faces=faces,
            color=(0.10, 0.42, 1.0, 1.0)
        )
        self._canvas.update()
        vispy.app.process_events()

    def __bool__(self):
        return not self._canvas._closed


def initialize_boids():
    """Create scattered initial positions so pheromone appears after grouping."""
    positions = np.random.uniform(-0.82, 0.82, size=(N, 2))

    angles = np.random.rand(N) * 2 * np.pi
    speeds = MIN_VEL + np.random.rand(N) * (MAX_VEL - MIN_VEL)
    velocities = np.column_stack((np.cos(angles), np.sin(angles))) * speeds[:, np.newaxis]
    return positions, velocities


def update_pheromones(
        positions,
        velocities,
        food_positions,
        danger_positions,
        food_reactions,
        group_contact_frames,
        food_layer,
        group_layer,
        trail_layer,
        danger_layer):
    """Deposit vector pheromone for groups, food discovery, trails, and danger."""
    neighbor_counts = local_neighbor_counts(positions)

    for i, position in enumerate(positions):
        if neighbor_counts[i] >= GROUP_PHEROMONE_MIN_NEIGHBORS:
            group_contact_frames[i] += 1
            if group_contact_frames[i] >= GROUP_PHEROMONE_REQUIRED_FRAMES:
                deposit_vector_pheromone(group_layer, position, velocities[i], GROUP_PHEROMONE_DEPOSIT)
                deposit_vector_pheromone(trail_layer, position, velocities[i], TRAIL_PHEROMONE_DEPOSIT)
        else:
            group_contact_frames[i] = max(0, group_contact_frames[i] - GROUP_CONTACT_DECAY)

        for food_index, food_pos in enumerate(food_positions):
            if np.linalg.norm(food_pos - position) < FOOD_DETECTION_RADIUS:
                departure_direction = food_departure_direction(position, velocities[i], food_pos)
                deposit_vector_pheromone(food_layer, position, departure_direction, FOOD_PHEROMONE_DEPOSIT)
                deposit_vector_pheromone(
                    trail_layer,
                    position,
                    departure_direction,
                    TRAIL_PHEROMONE_DEPOSIT * FOOD_DEPARTURE_TRAIL_MULTIPLIER
                )
                food_reactions[food_index] += 1

        for danger_pos in danger_positions:
            if np.linalg.norm(danger_pos - position) < DANGER_DETECTION_RADIUS:
                deposit_vector_pheromone(danger_layer, position, velocities[i], ALARM_PHEROMONE_DEPOSIT)

    for food_index in range(len(food_positions)):
        if food_reactions[food_index] >= FOOD_REACTION_LIMIT:
            relocate_food(food_positions, food_reactions, food_index)

    diffuse_vector_layer(food_layer)
    diffuse_vector_layer(group_layer)
    diffuse_vector_layer(trail_layer)
    diffuse_vector_layer(danger_layer)


def step(positions, velocities, food_positions, danger_positions,
         food_layer, group_layer, trail_layer, danger_layer):
    """Advance Boids by one frame using Boids and pheromone forces."""
    new_velocities = velocities.copy()
    for i in range(N):
        force = np.zeros(2)
        force += boids_force(i, positions, velocities)
        force += wander_force(velocities[i])
        force += pheromone_force(positions[i], velocities[i], food_layer, group_layer, trail_layer, danger_layer)
        force += food_and_danger_force(positions[i], food_positions, danger_positions)
        force += boundary_force(positions[i])
        new_velocities[i] = limit_velocity(velocities[i] + force)

    positions += new_velocities
    positions[:] = np.clip(positions, WORLD_MIN, WORLD_MAX)
    velocities[:] = new_velocities


def main():
    """Run the Pheromone Boids simulation."""
    positions, velocities = initialize_boids()
    food_positions = FOOD_POSITIONS.copy()
    food_reactions = np.zeros(len(food_positions), dtype=int)
    group_contact_frames = np.zeros(N, dtype=int)
    danger_positions = DANGER_POSITIONS.copy()
    danger_velocities = initialize_danger_velocities(danger_positions)
    food_pheromone = create_pheromone_layer()
    group_pheromone = create_pheromone_layer()
    trail_pheromone = create_pheromone_layer()
    danger_pheromone = create_pheromone_layer()
    visualizer = PheromoneBoidsVisualizer()

    while visualizer:
        move_danger_positions(danger_positions, danger_velocities)
        update_pheromones(
            positions,
            velocities,
            food_positions,
            danger_positions,
            food_reactions,
            group_contact_frames,
            food_pheromone,
            group_pheromone,
            trail_pheromone,
            danger_pheromone
        )
        step(
            positions,
            velocities,
            food_positions,
            danger_positions,
            food_pheromone,
            group_pheromone,
            trail_pheromone,
            danger_pheromone
        )
        visualizer.update(
            positions,
            velocities,
            food_positions,
            danger_positions,
            food_pheromone,
            group_pheromone,
            trail_pheromone,
            danger_pheromone
        )


if __name__ == '__main__':
    main()
