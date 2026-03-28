# Inverse Kinematics (FABRIK)

Forward And Backward Reaching Inverse Kinematics solver operating on `Actor.Bone` chains.

**File:** `ai4animation/IK/FABRIK.py`

---

## Overview

FABRIK is an iterative IK algorithm that solves for joint positions along a bone chain to reach a target position and rotation. It works by alternating forward and backward passes along the chain, adjusting joint positions to satisfy distance constraints.

<video controls autoplay loop muted width="100%">
  <source src="../../assets/videos/IK.mp4" type="video/mp4">
</video>

---

## API

### Constructor

```python
from ai4animation import FABRIK

ik = FABRIK(source_bone, target_bone)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `source_bone` | `Actor.Bone` | Start of the bone chain (e.g., shoulder) |
| `target_bone` | `Actor.Bone` | End of the bone chain (e.g., wrist) |

The constructor automatically builds the bone chain between the source and target bones.

### Solve

```python
ik.Solve(position, rotation, max_iterations=10, threshold=0.001)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `position` | `ndarray [3]` | Target world position |
| `rotation` | `ndarray [3, 3]` | Target world rotation |
| `max_iterations` | `int` | Maximum solver iterations |
| `threshold` | `float` | Convergence distance threshold |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `Bones` | `List[Actor.Bone]` | The bone chain from source to target |

### Internal Methods

| Method | Description |
|--------|-------------|
| `_backward_pass()` | Adjusts positions from end-effector to root |
| `_forward_pass()` | Adjusts positions from root to end-effector |
| `_assign()` | Writes solved transforms back to bones |

---

## Algorithm

1. **Backward pass**: Start at the target position, move each joint toward its parent while preserving bone lengths
2. **Forward pass**: Start at the root (fixed), move each joint toward its child while preserving bone lengths
3. **Repeat** until the end-effector is within the threshold distance of the target, or max iterations reached
4. **Assign**: Write the solved positions and rotations back to the actor's bones

---

## Example

From `Demos/InverseKinematics/Program.py`:

```python
import os
from ai4animation import Actor, AI4Animation, FABRIK, Vector3


class Program:
    def Start(self):
        actor = AI4Animation.Scene.AddEntity("Actor")
        self.Actor = actor.AddComponent(
            Actor, os.path.join(ASSETS_PATH, "Model.glb"), bone_names
        )

        self.IK = FABRIK(
            self.Actor.GetBone("LeftShoulder"),
            self.Actor.GetBone("LeftWrist"),
        )

        self.Target = AI4Animation.Scene.AddEntity("Target")
        self.Target.SetPosition(
            self.Actor.GetBone("LeftWrist").GetPosition()
        )

        self.Pose = self.Actor.GetTransforms()

    def Update(self):
        self.Actor.SetTransforms(self.Pose)
        self.IK.Solve(
            self.Target.GetPosition(),
            self.Target.GetRotation(),
            max_iterations=10,
            threshold=0.001,
        )
        self.Actor.SyncToScene(self.IK.Bones)

    def GUI(self):
        self.Target.DrawHandle()


if __name__ == "__main__":
    AI4Animation(Program(), mode=AI4Animation.Mode.STANDALONE)
```

This creates a draggable target handle — moving it drives the left arm via IK in real-time.
