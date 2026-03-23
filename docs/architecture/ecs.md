# Entity-Component-System

AI4AnimationPy uses an Entity-Component-System (ECS) pattern inspired by game engine design. This provides a flexible, composable architecture for building interactive applications.

---

## Core Concepts

### Scene

The `Scene` is the world manager. It holds all entities and stores their transforms as a **contiguous tensor**, enabling efficient batch operations.

```python
from ai4animation import AI4Animation

scene = AI4Animation.Scene
entity = scene.AddEntity("MyObject", position=Vector3.Create(0, 1, 0))
```

**Key responsibilities:**

- Manages the `Entities` list
- Stores `Transforms` as a contiguous `[N, 4, 4]` tensor
- Stores `Scales` as a `[N, 3]` tensor
- Drives per-entity lifecycle callbacks (`Update`, `Draw`, `GUI`)
- Provides batch transform operations (`SetTransforms`, `GetTransforms`)

### Entity

An `Entity` is a node in the scene graph. Each entity has:

- A unique **index** into the scene's transform tensor
- A **name** for identification
- A **parent-child hierarchy** with automatic forward kinematics propagation
- A dictionary of attached **components**

```python
root = AI4Animation.Scene.AddEntity("Root", position=Vector3.Create(0, 1, 0))
child = AI4Animation.Scene.AddEntity("Child", position=Vector3.Create(0, 2, 1), parent=root)
```

**Key operations:**

| Method | Description |
|--------|-------------|
| `SetTransform(transform)` | Sets the 4×4 transform, propagates to successors |
| `SetPosition(position)` | Sets position component of transform |
| `SetRotation(rotation)` | Sets rotation component of transform |
| `AddComponent(ComponentClass, *params)` | Attaches a component instance |
| `GetComponent(ComponentClass)` | Retrieves attached component by type |
| `FindChild(name)` | Recursively searches children by name |
| `GetChain(source, target)` | Returns entity chain between two nodes |
| `SetParent(parent)` | Reparents entity, updates successor lists |

### Component

`Component` is the abstract base class for all behaviors. Subclass it and override lifecycle hooks:

```python
from ai4animation.Components.Component import Component

class MyBehavior(Component):
    def Start(self, params):
        self.speed = params[0]

    def Update(self):
        # Called every frame
        pass

    def Draw(self):
        # Called every frame (standalone only, inside render pass)
        pass

    def GUI(self):
        # Called every frame (standalone only, after render)
        pass

    def Standalone(self):
        # Called once after Start (standalone only)
        pass
```

---

## Transform Storage Design

Entity transforms are stored in a **contiguous tensor** managed by the `Scene`:

```
Scene.Transforms: ndarray [N, 4, 4]  # N = total entities
Scene.Scales:     ndarray [N, 3]
```

Each entity has an `Index` that maps directly into this tensor. This design enables:

- **Batch operations** — set/get transforms for multiple entities at once
- **Efficient FK propagation** — when a parent moves, all successors are updated via tensor operations
- **GPU compatibility** — the tensor can be placed on GPU via PyTorch backend

---

## Parent-Child Hierarchy

Entities form a tree via parent-child relationships:

```python
root = scene.AddEntity("Root")
hip  = scene.AddEntity("Hip", parent=root)
leg  = scene.AddEntity("Leg", parent=hip)
foot = scene.AddEntity("Foot", parent=leg)
```

When a parent's transform changes, **all successors** (children, grandchildren, etc.) are automatically updated via forward kinematics propagation.

Each entity maintains:

- `Parent` — direct parent entity (or `None` for roots)
- `Children` — list of direct child entities
- `Successors` — flat list of **all** descendant indices (precomputed for efficient FK)

---

## Example: ECS in Practice

From `Demos/ECS/Program.py`:

```python
from ai4animation import AI4Animation, Component, Tensor, Time, Vector3


class Program:
    def Start(self):
        self.Root = AI4Animation.Scene.AddEntity(
            "Root", position=Vector3.Create(0, 1.0, 0)
        )
        self.Child = AI4Animation.Scene.AddEntity(
            "Child", position=Vector3.Create(0, 2.0, 1.0), parent=self.Root
        )

        cube = AI4Animation.Standalone.Primitives.CreateCube(
            "Cube", position=Vector3.Create(-1, 2, -2.5)
        )
        cube.AddComponent(self.Oscillator, 0.5, 1)

    class Oscillator(Component):
        def Start(self, params):
            self.Amplitude = params[0]
            self.Frequency = params[1]
            self.Position = self.Entity.GetPosition().copy()

        def Update(self):
            self.Entity.SetPosition(
                self.Position
                + Vector3.Create(
                    0.0,
                    self.Amplitude
                    * Tensor.Sin(
                        self.Frequency * 360.0 * Time.TotalTime, inDegrees=True
                    ),
                    0.0,
                )
            )


if __name__ == "__main__":
    AI4Animation(Program(), mode=AI4Animation.Mode.STANDALONE)
```

This demonstrates:

1. Creating entities with parent-child relationships
2. Defining a custom `Component` (Oscillator) with `Start` and `Update` hooks
3. Attaching the component to an entity with parameters
4. Using `Time.TotalTime` for animation
