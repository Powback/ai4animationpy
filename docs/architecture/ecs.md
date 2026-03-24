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
