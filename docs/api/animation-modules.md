# Animation Modules

Pluggable analysis modules that compute motion features from a `Motion` object. Each module can provide `Callback`, `Draw`, and `GUI` hooks for visualization and processing.

**Base File:** `ai4animation/Animation/Module.py`

<video controls autoplay loop muted width="100%">
  <source src="../../assets/videos/MotionEditor.mp4" type="video/mp4">
</video>

---

## Module Base Class

All animation modules inherit from `Module`:

```python
from ai4animation.Animation.Module import Module

class MyModule(Module):
    def __init__(self, motion):
        super().__init__(motion)
        # Precompute features from motion data

    def Callback(self, editor):
        # Called each frame during MotionEditor playback
        pass

    def Draw(self, editor):
        # Render debug visualization
        if Module.Visualize[MyModule]:
            pass

    def GUI(self, editor):
        # UI elements
        pass
```

Modules are attached to `Motion` objects via lambda factories when creating a `Dataset`:

```python
Dataset(path, [
    lambda x: RootModule(x, ...),
    lambda x: ContactModule(x, ...),
])
```

---

## Built-in Modules

### RootModule

**File:** `ai4animation/Animation/RootModule.py`

Computes the root trajectory from hip and shoulder landmark positions.

**Constructor:**

```python
RootModule(motion, hip_name, left_hip, right_hip, left_shoulder, right_shoulder)
```

**Key Outputs:**

- Root transforms (position + rotation)
- Root velocities
- Delta vectors for trajectory prediction

**How it works:** The root position is derived from the midpoint of hip landmarks projected onto the ground plane. The root rotation is computed from the forward direction defined by the hip-shoulder cross product.

### ContactModule

**File:** `ai4animation/Animation/ContactModule.py`

Detects foot contacts using height and velocity thresholds.

**Constructor:**

```python
ContactModule(motion, contacts)
# contacts: list of (bone_name, height_threshold, velocity_threshold)
```

**Example:**

```python
ContactModule(motion, [
    ("LeftAnkle", 0.1, 0.25),
    ("LeftBall", 0.05, 0.25),
    ("RightAnkle", 0.1, 0.25),
    ("RightBall", 0.05, 0.25),
])
```

**Key Outputs:** Binary contact labels per bone per frame.

**How it works:** A bone is considered "in contact" when its height is below the height threshold AND its velocity magnitude is below the velocity threshold.

### MotionModule

**File:** `ai4animation/Animation/MotionModule.py`

Wraps full-body bone trajectories with Gaussian smoothing for temporal consistency.

**Constructor:**

```python
MotionModule(motion)
```

**Key Outputs:** Per-bone transforms and velocities over a time window, smoothed with Gaussian kernels.

### GuidanceModule

**File:** `ai4animation/Animation/GuidanceModule.py`

Computes averaged pose guidance in root space, providing a coarse representation of the target pose.

**Constructor:**

```python
GuidanceModule(motion)
```

**Key Outputs:** Root-relative bone positions averaged over a window, used as guidance signals for neural network control.

### TrackingModule

**File:** `ai4animation/Animation/TrackingModule.py`

Tracks 3-point head and wrist positions, useful for VR/XR applications.

**Constructor:**

```python
TrackingModule(motion)
```

**Key Outputs:** Tracker transforms and velocities for head and both wrists.

---

## Nested Series Classes

Each module defines a nested `Series` class that extends `TimeSeries`. The `Series` stores precomputed temporal data and provides a `Draw()` method for visualization.

```python
class RootModule(Module):
    class Series(TimeSeries):
        def __init__(self, ...):
            super().__init__(start, end, samples)
            self.Transforms = ...
            self.Velocities = ...

        def Draw(self):
            # Render trajectory visualization
            pass
```

---

## Module Summary

| Module | Purpose | Key Outputs |
|--------|---------|-------------|
| **RootModule** | Root trajectory from hip/shoulder landmarks | Root transforms, velocities, delta vectors |
| **ContactModule** | Foot contact detection via height + velocity | Binary contact labels per bone |
| **MotionModule** | Full-body trajectories with Gaussian smoothing | Per-bone transforms/velocities over time |
| **GuidanceModule** | Averaged pose guidance in root space | Root-relative bone positions |
| **TrackingModule** | 3-point head + wrist tracking | Tracker transforms/velocities |
