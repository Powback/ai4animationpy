# Execution Modes

AI4AnimationPy supports three execution modes, selectable when creating the engine instance. Each mode determines how the update loop runs and whether rendering is enabled.

---

## Standalone Mode

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

## Headless Mode

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

## Manual Mode

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

**When to use:** Integration with external systems, unit testing, custom simulation loops, Jupyter notebooks.

---

## Comparison

| Feature | Standalone | Headless | Manual |
|---------|-----------|----------|--------|
| Window | ✅ | ❌ | ❌ |
| Rendering | ✅ | ❌ | ❌ |
| `Draw()` / `GUI()` | ✅ | ❌ | ❌ |
| Automatic loop | ✅ | ✅ | ❌ |
| Frame timing | Raylib | Wall-clock | Caller-provided |
| Camera / Skinned Mesh | ✅ | ❌ | ❌ |
| Server-compatible | ❌ | ✅ | ✅ |
