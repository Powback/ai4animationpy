# Tutorial: Custom Component

This tutorial walks through creating a custom `Component` that can be attached to any entity in the scene.

---

## Step 1: Subclass Component

Every custom behavior extends the `Component` base class:

```python
from ai4animation.Components.Component import Component


class Oscillator(Component):
    def Start(self, params):
        # Called once when the component is attached
        self.Amplitude = params[0]
        self.Frequency = params[1]
        self.Position = self.Entity.GetPosition().copy()
```

The `Start(params)` method receives a tuple of parameters passed during attachment.

---

## Step 2: Implement Lifecycle Hooks

Override any combination of lifecycle methods:

```python
from ai4animation import Tensor, Time, Vector3


class Oscillator(Component):
    def Start(self, params):
        self.Amplitude = params[0]
        self.Frequency = params[1]
        self.Position = self.Entity.GetPosition().copy()

    def Update(self):
        # Called every frame — game logic goes here
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

    def Draw(self):
        # Called every frame in standalone mode, inside render pass
        # Use AI4Animation.Draw to render debug visuals
        pass

    def GUI(self):
        # Called every frame in standalone mode, after rendering
        # Use for UI overlays
        pass

    def Standalone(self):
        # Called once after Start in standalone mode
        # Use for GUI setup, camera configuration
        pass
```

## Step 3: Attach to an Entity

```python
from ai4animation import AI4Animation, Vector3


class Program:
    def Start(self):
        cube = AI4Animation.Standalone.Primitives.CreateCube(
            "MyCube", position=Vector3.Create(0, 2, 0)
        )
        cube.AddComponent(Oscillator, 0.5, 1.0)  # amplitude=0.5, frequency=1.0
```

The parameters after the component class are passed as the `params` tuple to `Start()`.

---

## Step 4: Access Entity Properties

Inside any lifecycle method, use `self.Entity` to access the owning entity:

```python
class MyComponent(Component):
    def Update(self):
        pos = self.Entity.GetPosition()
        rot = self.Entity.GetRotation()
        name = self.Entity.Name

        # Access parent
        parent = self.Entity.Parent

        # Find sibling components
        actor = self.Entity.GetComponent(Actor)

        # Access the scene
        scene = AI4Animation.Scene
```

---

## Complete Example

From `Demos/ECS/Program.py` — two custom behaviors (oscillating position and pulsing scale):

```python
from ai4animation import AI4Animation, Component, Tensor, Time, Vector3


class Program:
    def Start(self):
        cube1 = AI4Animation.Standalone.Primitives.CreateCube(
            "Cube1", position=Vector3.Create(-1, 2, -2.5)
        )
        cube1.AddComponent(self.BounceComponent, 0.5, 1)

        cube2 = AI4Animation.Standalone.Primitives.CreateCube(
            "Cube2", position=Vector3.Create(3, 2, -2.5)
        )
        cube2.AddComponent(self.PulseComponent, 0.25, 1)

    class BounceComponent(Component):
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

    class PulseComponent(Component):
        def Start(self, params):
            self.Amplitude = params[0]
            self.Frequency = params[1]
            self.Scale = self.Entity.GetScale().copy()

        def Update(self):
            self.Entity.SetScale(
                self.Scale
                + self.Amplitude
                * Tensor.Sin(
                    self.Frequency * 360.0 * Time.TotalTime, inDegrees=True
                )
                * Vector3.One()
            )


if __name__ == "__main__":
    AI4Animation(Program(), mode=AI4Animation.Mode.STANDALONE)
```
