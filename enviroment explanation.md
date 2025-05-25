# Simple Spread Scenario

This document describes the implementation and structure of the `simple_spread` scenario, which is a specific environment setup derived from a base scenario used in multi-agent environments such as Multi-Agent Particle Environments (MPE).

## Base Scenario Structure

Every scenario must inherit from the base scenario class and implement the following methods:

```python
class BaseScenario(object):
    def make_world(self):
        raise NotImplementedError()

    def reset_world(self, world):
        raise NotImplementedError()

    def info(self, agent, world):
        return {}
```

## Implementation: `simple_spread`

### `make_world(self, args)`

This method is responsible for initializing the world, setting its properties, adding agents and landmarks, and calling `reset_world` to configure initial conditions.

```python
def make_world(self, args):
    world = World()
    world.world_length = args["episode_length"]
    world.dim_c = 2
    world.num_agents = args["num_agents"]
    world.num_landmarks = args["num_landmarks"]
    world.collaborative = True

    world.agents = [Agent() for _ in range(world.num_agents)]
    for i, agent in enumerate(world.agents):
        agent.name = f'agent {i}'
        agent.collide = True
        agent.silent = True
        agent.size = 0.15

    world.landmarks = [Landmark() for _ in range(world.num_landmarks)]
    for i, landmark in enumerate(world.landmarks):
        landmark.name = f'landmark {i}'
        landmark.collide = False
        landmark.movable = False

    self.reset_world(world)
    return world
```

### Key Parameters

* `world_length`: Number of steps in an episode.
* `dim_c = 2`: Communication vector size (usually 2D: x and y). However, if agents are silent, this is unused.
* `collide = True`: Enables physical collision detection for agents.
* `silent = True`: Disables communication. As a result, the communication state is set to zero vectors during simulation.

Example from `mpe_core.py`:

```python
def update_agent_state(self, agent):
    if agent.silent:
        agent.state.c = np.zeros(self.dim_c)
    else:
        noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
        agent.state.c = agent.action.c + noise
```

### Notes

* Communication (`dim_c`) is set up even when unused. This might be relevant if communication is later enabled.
* Landmarks are immovable and non-collidable, often used as goal targets for agents.

This setup provides the fundamental layout for collaborative multi-agent tasks where agents learn to navigate the environment without communication and must avoid collisions while possibly achieving proximity-based goals.

# Additional Implementation Details for `simple_spread`

This section expands on the remaining methods of the `simple_spread` scenario, focusing on world initialization, rewards, and performance benchmarking.

## `reset_world(self, world)`

This method sets the initial conditions of the world:

```python
def reset_world(self, world):
    world.assign_agent_colors()
    world.assign_landmark_colors()

    for agent in world.agents:
        agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
        agent.state.p_vel = np.zeros(world.dim_p)
        agent.state.c = np.zeros(world.dim_c)

    for landmark in world.landmarks:
        landmark.state.p_pos = 0.8 * np.random.uniform(-1, +1, world.dim_p)
        landmark.state.p_vel = np.zeros(world.dim_p)
```

### Color Assignment Helpers

* **`assign_agent_colors()`**: Assigns distinct colors based on agent roles:

  ```python
  dummy_colors = [(0.25, 0.75, 0.25)]  # green
  adv_colors = [(0.75, 0.25, 0.25)]    # red
  good_colors = [(0.25, 0.25, 0.75)]   # blue
  ```

  These categories (dummy, adversary, good) may or may not be used depending on the specific scenario.

* **`assign_landmark_colors()`**: All landmarks are given a standard gray color:

  ```python
  landmark.color = np.array([0.25, 0.25, 0.25])
  ```

## `reward(self, agent, world)`

This function determines the agent's reward each step:

```python
def reward(self, agent, world):
    rew = 0
    for l in world.landmarks:
        dists = [np.linalg.norm(a.state.p_pos - l.state.p_pos) for a in world.agents]
        rew -= min(dists)

    if agent.collide:
        for a in world.agents:
            if self.is_collision(a, agent):
                rew -= 1
    return rew
```

### Explanation:

* Reward is the negative of the sum of minimum distances between agents and landmarks.
* An additional penalty of -1 is applied for each collision this agent is involved in.

## `is_collision(self, agent1, agent2)`

Checks if two agents are colliding:

```python
def is_collision(self, agent1, agent2):
    delta_pos = agent1.state.p_pos - agent2.state.p_pos
    dist = np.linalg.norm(delta_pos)
    dist_min = agent1.size + agent2.size
    return dist < dist_min
```

## `benchmark_data(self, agent, world)`

Returns a tuple of detailed performance statistics:

```python
def benchmark_data(self, agent, world):
    rew = 0
    collisions = 0
    occupied_landmarks = 0
    min_dists = 0
    for l in world.landmarks:
        dists = [np.linalg.norm(a.state.p_pos - l.state.p_pos) for a in world.agents]
        min_dists += min(dists)
        rew -= min(dists)
        if min(dists) < 0.1:
            occupied_landmarks += 1

    if agent.collide:
        for a in world.agents:
            if self.is_collision(a, agent):
                rew -= 1
                collisions += 1

    return (rew, collisions, min_dists, occupied_landmarks)
```

### Return Values:

* **rew**: Same reward value as returned by `reward()`.
* **collisions**: Number of collisions for this agent.
* **min\_dists**: Sum of the closest distances from all landmarks to any agent.
* **occupied\_landmarks**: Number of landmarks with an agent closer than 0.1 units.

# Observation in `simple_spread`

In multi-agent environments, the observation function defines what each agent perceives about the world at every step. The observation is typically a vector that contains relevant information an agent can use for decision-making.

## `observation(self, agent, world)`

This method constructs an observation vector for a given agent:

```python
def observation(self, agent, world):
    entity_pos = []
    for entity in world.landmarks:
        entity_pos.append(entity.state.p_pos - agent.state.p_pos)

    entity_color = []
    for entity in world.landmarks:
        entity_color.append(entity.color)

    comm = []
    other_pos = []
    other_vel = []
    for other in world.agents:
        if other is agent:
            continue
        comm.append(other.state.c)
        other_pos.append(other.state.p_pos - agent.state.p_pos)
        other_vel.append(other.state.p_vel)

    return np.concatenate([
        agent.state.p_vel,
        agent.state.p_pos,
        *entity_pos,
        *other_pos,
        *comm
    ])
```

## Explanation of Components

The resulting observation vector includes:

* **Self velocity (`agent.state.p_vel`)**: The agent’s current velocity.
* **Self position (`agent.state.p_pos`)**: The agent’s position in the world.
* **Relative positions of landmarks**: Each landmark’s position relative to the observing agent.
* **Relative positions of other agents**: Positions of all other agents, relative to the observing agent.
* **Communication vectors (`other.state.c`)**: Messages sent by other agents (if communication is enabled).

Note: `entity_color` and `other_vel` are computed but not used in the final return value. They could be added for richer observations if needed.

## Use in Training

This observation is passed to the agent’s policy network or decision model. The agent uses it to choose actions that maximize cumulative reward. In `simple_spread`, communication vectors are typically zero (since `silent = True`), but the structure supports scenarios with communication enabled.

### Notes
A `world()` class is just a structure that represents the components of an environment. but this class contain some important functions for mpe enviroments in general (not just simple spread).
# Physics and Dynamics in `simple_spread`

This section describes core physics and simulation functions used by the environment to manage entity interactions, forces, and state updates.

---

## `calculate_distances(self)`

Pre-computes pairwise position differences and collision thresholds for all entities.

```python
def calculate_distances(self):
    if self.cached_dist_vect is None:
        # initialize distance tensor and minimum collision distances
        n = len(self.entities)
        self.cached_dist_vect = np.zeros((n, n, self.dim_p))
        self.min_dists = np.zeros((n, n))
        for ia in range(n):
            for ib in range(ia+1, n):
                min_dist = self.entities[ia].size + self.entities[ib].size
                self.min_dists[ia, ib] = min_dist
                self.min_dists[ib, ia] = min_dist
    # update relative position vectors
    for ia, a in enumerate(self.entities):
        for ib in range(ia+1, len(self.entities)):
            delta = a.state.p_pos - self.entities[ib].state.p_pos
            self.cached_dist_vect[ia, ib] = delta
            self.cached_dist_vect[ib, ia] = -delta
    # magnitudes and collision mask
    self.cached_dist_mag = np.linalg.norm(self.cached_dist_vect, axis=2)
    self.cached_collisions = (self.cached_dist_mag <= self.min_dists)
```

**Purpose:**

* Speeds up collision checks by caching distances and thresholds.
* `cached_collisions[i,j]` is `True` if entities `i` and `j` overlap.

---

## `step(self)`

Advances the world state by one time step, applying actions and forces, then updating all entities.

```python
def step(self):
    self.world_step += 1
    # scripted agents choose actions
    for agent in self.scripted_agents:
        agent.action = agent.action_callback(agent, self)
    # gather forces
    p_force = [None] * len(self.entities)
    p_force = self.apply_action_force(p_force)
    p_force = self.apply_environment_force(p_force)
    # integrate motion
    self.integrate_state(p_force)
    # update agent communication state
    for agent in self.agents:
        self.update_agent_state(agent)
    # optionally cache distances
    if self.cache_dists:
        self.calculate_distances()
```

**Flow:**

1. Collect actions from scripted agents.
2. Compute action-based and environmental forces.
3. Integrate physical state (positions & velocities).
4. Update agent-specific state (e.g., communication).
5. Update distance cache if enabled.

---

## `apply_action_force(self, p_force)`

Applies each agent’s control forces based on its action and noise.

```python
def apply_action_force(self, p_force):
    for i, agent in enumerate(self.agents):
        if agent.movable:
            noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
            # F = mass * a * action + noise
            coeff = agent.mass * (agent.accel or 1.0)
            p_force[i] = coeff * agent.action.u + noise
    return p_force
```

**Key Points:**

* Only movable agents generate forces.
* Action noise simulates imperfect actuation.

---

## `apply_environment_force(self, p_force)`

Adds collision and wall forces to the force accumulator.

```python
def apply_environment_force(self, p_force):
    # inter-entity collisions
    for a, ea in enumerate(self.entities):
        for b, eb in enumerate(self.entities):
            if b <= a: continue
            f_a, f_b = self.get_entity_collision_force(a, b)
            if f_a is not None:
                p_force[a] = (p_force[a] or 0.0) + f_a
            if f_b is not None:
                p_force[b] = (p_force[b] or 0.0) + f_b
        # wall collisions
        if ea.movable:
            for wall in self.walls:
                wf = self.get_wall_collision_force(ea, wall)
                if wf is not None:
                    p_force[a] = (p_force[a] or 0.0) + wf
    return p_force
```

**Details:**

* Calls `get_entity_collision_force` for every entity pair.
* Applies wall responses via `get_wall_collision_force`.

---

## `integrate_state(self, p_force)`

Updates entities’ velocities and positions using Euler integration.

```python
def integrate_state(self, p_force):
    for i, ent in enumerate(self.entities):
        if not ent.movable:
            continue
        ent.state.p_vel *= (1 - self.damping)
        if p_force[i] is not None:
            ent.state.p_vel += (p_force[i] / ent.mass) * self.dt
        # enforce max speed
        if ent.max_speed is not None:
            speed = np.linalg.norm(ent.state.p_vel)
            if speed > ent.max_speed:
                ent.state.p_vel = ent.state.p_vel / speed * ent.max_speed
        ent.state.p_pos += ent.state.p_vel * self.dt
```

**Notes:**

* Applies damping to simulate friction.
* Clamps velocity to `max_speed`.

---

## `update_agent_state(self, agent)`

Updates an agent’s communication vector.

```python
def update_agent_state(self, agent):
    if agent.silent:
        agent.state.c = np.zeros(self.dim_c)
    else:
        noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
        agent.state.c = agent.action.c + noise
```

**Behavior:**

* Silent agents have zero communication.
* Otherwise, communication includes optional noise.

---

## `get_entity_collision_force(self, ia, ib)`

Computes a soft collision force between two entities.

```python
def get_entity_collision_force(self, ia, ib):
    A, B = self.entities[ia], self.entities[ib]
    if not (A.collide and B.collide) or (not A.movable and not B.movable) or A is B:
        return [None, None]
    # distance and minimum threshold
    if self.cache_dists:
        delta = self.cached_dist_vect[ia, ib]
        dist, dmin = self.cached_dist_mag[ia, ib], self.min_dists[ia, ib]
    else:
        delta = A.state.p_pos - B.state.p_pos
        dist = np.linalg.norm(delta)
        dmin = A.size + B.size
    # penetration via soft margin
    k = self.contact_margin
    penetration = np.logaddexp(0, -(dist - dmin)/k) * k
    force = self.contact_force * delta/dist * penetration
    # split force by mass ratio
    if A.movable and B.movable:
        ratio = B.mass / A.mass
        fA = ratio * force
        fB = -force / ratio
    else:
        fA = force if A.movable else None
        fB = -force if B.movable else None
    return [fA, fB]
```

**Mechanics:**

* Uses a smooth log-exponential margin to compute penetration.
* Distributes forces based on mass ratios.

---

## `get_wall_collision_force(self, entity, wall)`

Calculates force when an entity contacts a wall boundary.

```python
def get_wall_collision_force(self, ent, wall):
    if ent.ghost and not wall.hard:
        return None
    # determine parallel/perpendicular axes
    prl = 0 if wall.orient=='H' else 1
    perp = 1-prl
    pos = ent.state.p_pos
    # skip if outside endpoints
    if pos[prl] < wall.endpoints[0]-ent.size or pos[prl] > wall.endpoints[1]+ent.size:
        return None
    # compute minimal distance and angle theta
    # ... (see full code for details)
    delta = pos[perp] - wall.axis_pos
    dist = abs(delta)
    k = self.contact_margin
    pen = np.logaddexp(0, -(dist - dmin)/k) * k
    mag = self.contact_force * delta/dist * pen
    force = np.zeros(2)
    force[perp] = np.cos(theta) * mag
    force[prl] = np.sin(theta) * abs(mag)
    return force
```

**Highlights:**

* Ghost entities pass through soft walls.
* Uses similar soft-penalty approach as entity collisions.
* Resolves wall width and endpoint effects via geometry.
