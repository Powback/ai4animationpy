# Lifecycle & Update Loop

AI4AnimationPy follows a game-engine-style lifecycle with distinct phases for initialization, updating, rendering, and GUI. Understanding this lifecycle is essential for writing correct programs.

---

## Program Template

Every program follows this pattern:

```python
from ai4animation import AI4Animation


class Program:
    def Start(self):
        # Called once at initialization
        # Create entities, load models, set up data
        pass

    def Standalone(self):
        # Called once after Start (standalone only)
        # Configure camera, create GUI elements
        pass

    def Update(self):
        # Called every frame
        # Game logic, animation, input handling
        pass

    def Draw(self):
        # Called every frame (standalone only, inside render pass)
        # Debug visualization, shape drawing
        pass

    def GUI(self):
        # Called every frame (standalone only, after render)
        # UI overlays, handles, text
        pass


if __name__ == "__main__":
    AI4Animation(Program(), mode=AI4Animation.Mode.STANDALONE)
```

---

## Lifecycle Phases

### 1. Initialization

When `AI4Animation(program, mode)` is called:

1. A `Scene` is created
2. If `mode == STANDALONE`, the Standalone renderer is loaded (Raylib window, camera, render pipeline)
3. `program.Start()` is called — set up entities, components, and data here
4. `program.Standalone()` is called (standalone mode only) — configure GUI, camera targets

### 2. Update

Called every frame before rendering:

```
AI4Animation.__UPDATE__()
├── program.Update()      # Your game logic
└── scene.Update()        # Iterates all entities
    └── entity.Update()   # For each entity
        └── component.Update()  # For each attached component
```

**Order:** Your program's `Update()` runs first, then the scene updates all entities and their components.

### 3. Draw

Called every frame inside the render pass (standalone only):

```
AI4Animation.__DRAW__()
├── program.Draw()        # Your rendering code
└── scene.Draw()          # Iterates all entities
    └── entity.Draw()     # For each entity
        └── component.Draw()  # For each attached component
```

Use `Draw()` to render debug visualizations, shapes, lines, etc. via `AI4Animation.Draw`.

### 4. GUI

Called every frame after rendering (standalone only):

```
AI4Animation.__GUI__()
├── program.GUI()         # Your UI overlays
└── scene.GUI()           # Iterates all entities
    └── entity.GUI()      # For each entity
        └── component.GUI()  # For each attached component
```

Use `GUI()` for immediate-mode UI elements like sliders, buttons, text overlays.

---

## Time

The `Time` module provides frame timing globals:

| Property | Type | Description |
|----------|------|-------------|
| `Time.DeltaTime` | `float` | Seconds elapsed since last frame |
| `Time.TotalTime` | `float` | Total seconds since engine start |
| `Time.Timescale` | `float` | Time multiplier (default: 1.0) |

```python
from ai4animation import Time

def Update(self):
    speed = 2.0
    distance = speed * Time.DeltaTime  # Frame-rate independent movement
    angle = 120 * Time.TotalTime       # Continuous rotation
```

!!! note
    `DeltaTime` is pre-multiplied by `Timescale`. Set `Time.Timescale = 0.5` to run at half speed.

---

## Execution Modes

AI4AnimationPy supports three execution modes, selectable when creating the engine instance. Each mode determines how the update loop runs and whether rendering is enabled.

![Order of Execution](../assets/images/OrderOfExecution.png){ width="100%" }

---

### Standalone Mode

**Windowed rendering** via Raylib with a full deferred rendering pipeline.

```python
from ai4animation import AI4Animation

class Program:
    def Start(self):
        pass
    def Update(self):
        pass
    def Draw(self):
        pass  # Called inside the render pass
    def GUI(self):
        pass  # Called after rendering for UI overlays

if __name__ == "__main__":
    AI4Animation(Program(), mode=AI4Animation.Mode.STANDALONE)
```

**Characteristics:**

- Opens a 1920×1080 window
- Runs the full render pipeline: deferred shading, shadow mapping, SSAO, bloom, FXAA
- Calls `Draw()` and `GUI()` every frame
- Camera, skinned mesh, and GUI systems are available
- Frame timing is driven by Raylib's `GetFrameTime()`

**When to use:** Development, visualization, debugging, demos.

---

### Headless Mode

**Infinite update loop** without rendering. No window is created.

```python
from ai4animation import AI4Animation

class Program:
    def Start(self):
        pass
    def Update(self):
        pass  # Called every iteration

if __name__ == "__main__":
    AI4Animation(Program(), mode=AI4Animation.Mode.HEADLESS)
```

**Characteristics:**

- No window or rendering
- `Draw()` and `GUI()` are never called
- Runs `Update()` in a tight loop as fast as possible
- Frame time is computed from wall-clock time

**When to use:** Server-side training, batch processing, data generation, CI/CD pipelines.

---

### Manual Mode

**Caller-controlled** update timing. The engine does not run its own loop.

```python
from ai4animation import AI4Animation

class Program:
    def Start(self):
        pass
    def Update(self):
        pass

engine = AI4Animation(Program(), mode=AI4Animation.Mode.MANUAL)

# Drive the update loop externally
for i in range(1000):
    AI4Animation.Update(deltaTime=1.0/60.0)
```

**Characteristics:**

- No automatic loop — you call `AI4Animation.Update(deltaTime)` yourself
- Full control over frame timing and step count
- No rendering (same as Headless in that regard)

**When to use:** Integration with external systems, custom simulation loops, Jupyter notebooks.

---

## Standalone Renderer

When running in standalone mode, the engine creates:

- A **Raylib window**
- A **Camera**
- A **RenderPipeline**
- A **Ground** plane entity

The render loop sequence per frame:

1. `GetFrameTime()` → compute delta time
2. `Update()` → game logic + scene update
3. `BeginDrawing()` → start render pass
4. `Draw()` → custom rendering
5. `GUI()` → UI overlays
6. `EndDrawing()` → present frame
