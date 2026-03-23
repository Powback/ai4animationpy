# Tutorial: Custom Animation Module

This tutorial shows how to create a custom animation `Module` that computes features from motion data and integrates with the `MotionEditor`.

---

## Step 1: Subclass Module

```python
from ai4animation.Animation.Module import Module


class MyFeatureModule(Module):
    def __init__(self, motion):
        super().__init__(motion)
        self.features = self._compute(motion)

    def _compute(self, motion):
        # Precompute features from motion data
        # This runs once when the module is created
        num_frames = motion.Frames.shape[0]
        features = []
        for i in range(num_frames):
            # Example: compute center of mass height
            positions = motion.Frames[i, :, :3, 3]  # [J, 3]
            com_height = positions[:, 1].mean()
            features.append(com_height)
        return features
```

---

## Step 2: Add Callback and Draw Hooks

```python
class MyFeatureModule(Module):
    def __init__(self, motion):
        super().__init__(motion)
        self.features = self._compute(motion)

    def _compute(self, motion):
        num_frames = motion.Frames.shape[0]
        return [motion.Frames[i, :, 1, 3].mean() for i in range(num_frames)]

    def Callback(self, editor):
        # Called each frame during MotionEditor playback
        # Access current timestamp via editor
        pass

    def Draw(self, editor):
        # Render debug visualization
        if Module.Visualize[MyFeatureModule]:
            # Draw only when visualization is enabled
            pass

    def GUI(self, editor):
        # Add UI elements
        pass
```

---

## Step 3: Attach to a Dataset

Modules are attached via lambda factories when creating a `Dataset`:

```python
from ai4animation import Dataset, RootModule

dataset = Dataset(
    "path/to/motions",
    [
        lambda x: RootModule(x, hip, l_hip, r_hip, l_shoulder, r_shoulder),
        lambda x: MyFeatureModule(x),  # Your custom module
    ],
)
```

The lambda receives the `Motion` object and must return a `Module` instance. This pattern allows modules to be lazily instantiated when a motion is loaded.

---

## Step 4: Add a Nested Series Class (Optional)

For temporal windowing, define a nested `Series` class extending `TimeSeries`:

```python
from ai4animation.Animation.TimeSeries import TimeSeries


class MyFeatureModule(Module):
    def __init__(self, motion):
        super().__init__(motion)
        self.features = self._compute(motion)

    def _compute(self, motion):
        return [motion.Frames[i, :, 1, 3].mean() for i in range(motion.Frames.shape[0])]

    class Series(TimeSeries):
        def __init__(self, start, end, samples):
            super().__init__(start, end, samples)
            self.Heights = []

        def Draw(self):
            # Render the series data
            pass
```

---

## Complete Example

```python
from ai4animation.Animation.Module import Module
from ai4animation import Dataset, MotionEditor, AI4Animation
import os


class CenterOfMassModule(Module):
    """Tracks the center of mass height over time."""

    def __init__(self, motion):
        super().__init__(motion)
        self.com_heights = []
        for i in range(motion.Frames.shape[0]):
            positions = motion.Frames[i, :, :3, 3]
            self.com_heights.append(float(positions[:, 1].mean()))

    def Callback(self, editor):
        pass

    def Draw(self, editor):
        if Module.Visualize[CenterOfMassModule]:
            pass


class Program:
    def Start(self):
        editor = AI4Animation.Scene.AddEntity("MotionEditor")
        editor.AddComponent(
            MotionEditor,
            Dataset(
                os.path.join(ASSETS_PATH, "Motions"),
                [
                    lambda x: CenterOfMassModule(x),
                ],
            ),
            os.path.join(ASSETS_PATH, "Model.glb"),
            bone_names,
        )

    def Update(self):
        pass


if __name__ == "__main__":
    AI4Animation(Program())
```
