# Demo Programs

Reference table of all demo programs included in the repository. Each demo illustrates specific framework concepts and can be run directly.

---

## Running Demos

From the repository root:

```bash
python Demos/<DemoName>/Program.py
```

Most demos run in **Standalone mode** (windowed rendering). Some can also run headless.

---

## Demo Index

| Demo | Description | Key Concepts | Mode |
|------|-------------|-------------|------|
| **[Empty](https://github.com/facebookresearch/ai4animationpy/tree/main/Demos/Empty)** | Minimal program — prints "Hello World" | Engine bootstrap, `Program` class | Headless |
| **[ECS](https://github.com/facebookresearch/ai4animationpy/tree/main/Demos/ECS)** | Entity hierarchy with custom components | `AddEntity`, `AddComponent`, parent-child, custom `Component` | Standalone |
| **[Actor](https://github.com/facebookresearch/ai4animationpy/tree/main/Demos/Actor)** | Load and display a character model | `Actor` component, GLB loading, `SyncFromScene` | Standalone |
| **[GLBLoading](https://github.com/facebookresearch/ai4animationpy/tree/main/Demos/GLBLoading)** | Load character from GLB format | `GLBImporter`, skinned mesh | Standalone |
| **[FBXLoading](https://github.com/facebookresearch/ai4animationpy/tree/main/Demos/FBXLoading)** | Load character from FBX format | `FBXImporter` | Standalone |
| **[BVHLoading](https://github.com/facebookresearch/ai4animationpy/tree/main/Demos/BVHLoading)** | Load motion from BVH format | `BVHImporter`, `Motion` | Standalone |
| **[MotionEditor](https://github.com/facebookresearch/ai4animationpy/tree/main/Demos/MotionEditor)** | Browse and play motion clips with timeline | `MotionEditor`, `Dataset`, animation modules | Standalone |
| **[InverseKinematics](https://github.com/facebookresearch/ai4animationpy/tree/main/Demos/InverseKinematics)** | Real-time IK with draggable target | `FABRIK` solver, `Actor.GetBone` | Standalone |
| **[Locomotion](https://github.com/facebookresearch/ai4animationpy/tree/main/Demos/Locomotion)** | Neural network driven locomotion | Full inference pipeline, `FeedTensor`/`ReadTensor`, IK | Standalone |
| **[AI/ToyExample](https://github.com/facebookresearch/ai4animationpy/tree/main/Demos/AI/ToyExample)** | Minimal training example | `MLP`, optimizer, scheduler, loss history | Standalone |
| **[AI/Autoencoder](https://github.com/facebookresearch/ai4animationpy/tree/main/Demos/AI/Autoencoder)** | Autoencoder training on motion data | VAE, motion features, `DataSampler` | Standalone |

---

## Demo Details

### Empty

The simplest possible program — demonstrates the basic `Program` class structure:

```python
from ai4animation import AI4Animation

class Program:
    def __init__(self, variable):
        self.Variable = variable

    def Start(self):
        print(self.Variable)

    def Update(self):
        return

if __name__ == "__main__":
    AI4Animation(Program("Hello World"), mode=AI4Animation.Mode.HEADLESS)
```

### ECS

Demonstrates the Entity-Component-System pattern with parent-child entity hierarchies, custom components (oscillating position, pulsing scale), and primitive shape creation.

### Actor

Loads a GLB character model, attaches it to a scene entity, and rotates it continuously using `Time.TotalTime`.

### MotionEditor

The most feature-rich demo — creates a `Dataset` with multiple animation modules (`RootModule`, `MotionModule`, `ContactModule`, `GuidanceModule`), provides a GUI timeline for scrubbing through motion clips, and visualizes module data.

### InverseKinematics

Creates a FABRIK solver on the left arm chain (shoulder → wrist), with a draggable target handle. Demonstrates real-time IK solving and `SyncToScene` for selective bone updates.

### Locomotion

Full neural network locomotion system with:

- Gamepad/keyboard input for velocity and direction
- Trained network inference at 10Hz
- Temporal blending of predictions
- Bone length restoration
- Leg IK for ground contact

### AI/ToyExample

Minimal training loop using a generator pattern:

- `MLP` network learning a simple function
- `AdamW` optimizer with `CyclicScheduler`
- `Plotting.LossHistory` for loss visualization
- Live function visualization comparing prediction vs ground truth
