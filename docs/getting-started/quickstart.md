# Quick Start

This guide walks through three progressively complex examples to get you started with AI4AnimationPy.

---

## Hello World — Empty Window

The simplest possible program creates an engine instance and runs the update loop:

```python
from ai4animation import AI4Animation


class Program:
    def __init__(self, variable):
        self.Variable = variable

    def Start(self):
        print(self.Variable)

    def Update(self):
        return

    def Draw(self):
        return

    def GUI(self):
        return


if __name__ == "__main__":
    AI4Animation(Program("Hello World"), mode=AI4Animation.Mode.HEADLESS)
```

This runs in **headless mode** (no window). The engine calls `Start()` once, then loops `Update()` indefinitely.

To open a **standalone window** instead:

```python
AI4Animation(Program("Hello World"), mode=AI4Animation.Mode.STANDALONE)
```

---

## Loading a Character

Load a 3D character model and display it in a window:

```python
from ai4animation import Actor, AI4Animation, Rotation, Time, Vector3

class Program:
    def Start(self):
        entity = AI4Animation.Scene.AddEntity("Actor")
        self.Actor = entity.AddComponent(
            Actor, model_path
        )
        self.Actor.Entity.SetPosition(Vector3.Create(0, 0, 0))

    def Update(self):
        self.Actor.Entity.SetRotation(Rotation.Euler(0, 120 * Time.TotalTime, 0))
        self.Actor.SyncFromScene()


if __name__ == "__main__":
    AI4Animation(Program(), mode=AI4Animation.Mode.STANDALONE)
```

**What this does:**

1. Creates a scene entity named `"Actor"`
2. Attaches an `Actor` component, loading a model file
3. Rotates the character every frame based on elapsed time
4. Calls `SyncFromScene()` to update internal bone transforms from the scene graph

<video controls autoplay loop muted width="100%">
  <source src="../../assets/videos/Actor.mp4" type="video/mp4">
</video>

---

## Importing Motion Data

AI4AnimationPy supports importing mesh, skin, and animation data from **GLB**, **FBX**, and **BVH** files. The internal motion format is `.npz`, which stores 7 dimensions (3D position + 4D quaternion) for each skeleton joint per frame.

<video controls autoplay loop muted width="100%">
  <source src="../../assets/videos/MocapImport.mp4" type="video/mp4">
</video>

### Loading Motion Files in Code

Use the `Motion` class to load directly from any supported format:

```python
from ai4animation import Motion

# Load from different formats
glb_motion = Motion.LoadFromGLB("character.glb", names=bone_names, floor=None)
fbx_motion = Motion.LoadFromFBX("character.fbx")
bvh_motion = Motion.LoadFromBVH("character.bvh", scale=0.01)

# Load from the internal NPZ format
npz_motion = Motion.LoadFromNPZ("character.npz")

# Save any motion to NPZ
glb_motion.SaveToNPZ("character")
```

!!! tip
    BVH files often use centimeters — pass `scale=0.01` to convert to meters.

!!! warning "FBX requires the Autodesk FBX SDK"
    FBX imports require the Autodesk FBX SDK Python bindings, which must be installed separately:

    1. Download the FBX SDK from [Autodesk FBX SDK](https://aps.autodesk.com/developer/overview/fbx-sdk)
    2. Download the FBX SDK Python Bindings
    3. Set the `FBXSDK_ROOT` environment variable to the FBX SDK install path
       (e.g. `$env:FBXSDK_ROOT = "C:\Program Files\Autodesk\FBX\FBX SDK\2020.3.9"`)
    4. Set the `FBXSDK_COMPILER` environment variable (e.g. `$env:FBXSDK_COMPILER="vs2022"`)
    5. `pip install --force-reinstall -v sip==6.6.2`
    6. `pip install .` (in the Python Bindings folder)

    GLB and BVH import work out of the box without any additional dependencies.

### Batch Conversion via CLI

To convert an entire directory of motion files to NPZ, use the built-in `convert` command:

```bash
convert --input_dir path/to/motions --output_dir path/to/output --skeleton Cranberry
```

| Argument | Description |
|----------|-------------|
| `--input_dir` | Directory containing GLB, FBX, or BVH files (searched recursively) |
| `--output_dir` | Output directory for NPZ files (default: `input_dir/NPZ`) |
| `--skeleton` | Optional skeleton definition for bone filtering: `Cranberry` or `Geno` |
| `--bvh_scale` | Scale factor for BVH position data (default: `0.01`) |


### Public Datasets

Several public motion capture datasets are compatible with the framework:

| Dataset | Character | Download |
|---------|-----------|----------|
| [Cranberry](https://github.com/sebastianstarke/AI4Animation) | Cranberry | [FBX & GLB](https://starke-consult.de/AI4Animation/SIGGRAPH_2024/Cranberry_Dataset.zip) |
| [100Style retargeted](https://github.com/orangeduck/100style-retarget) | Geno | [BVH](https://theorangeduck.com/media/uploads/Geno/100style-retarget/bvh.zip) / [FBX](https://theorangeduck.com/media/uploads/Geno/100style-retarget/fbx.zip) |
| [LaFan](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) | Ubisoft LaFan | [BVH](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip) |
| [LaFan resolved](https://github.com/orangeduck/lafan1-resolved) | Geno | [BVH](https://theorangeduck.com/media/uploads/Geno/lafan1-resolved/bvh.zip) / [FBX](https://theorangeduck.com/media/uploads/Geno/lafan1-resolved/fbx.zip) |
| [ZeroEggs retargeted](https://github.com/orangeduck/zeroeggs-retarget) | Geno | [BVH](https://theorangeduck.com/media/uploads/Geno/zeroeggs-retarget/bvh.zip) / [FBX](https://theorangeduck.com/media/uploads/Geno/zeroeggs-retarget/fbx.zip) |
| [Motorica retargeted](https://github.com/orangeduck/motorica-retarget) | Geno | [BVH](https://theorangeduck.com/media/uploads/Geno/motorica-retarget/bvh.zip) / [FBX](https://theorangeduck.com/media/uploads/Geno/motorica-retarget/fbx.zip) |
| [NSM](https://github.com/sebastianstarke/AI4Animation/tree/master/AI4Animation/SIGGRAPH_Asia_2019) | Anubis | [BVH](https://starke-consult.de/AI4Animation/SIGGRAPH_Asia_2019/MotionCapture.zip) |
| [MANN](https://github.com/sebastianstarke/AI4Animation/tree/master/AI4Animation/SIGGRAPH_2018) | Dog | [BVH](https://starke-consult.de/AI4Animation/SIGGRAPH_2018/MotionCapture.zip) |

---

## Playing Motion Data

Use the `MotionEditor` component to browse and play motion clips from NPZ files:

```python
from ai4animation import (
    AI4Animation,
    ContactModule,
    Dataset,
    MotionEditor,
    MotionModule,
    GuidanceModule,
    RootModule,
)

class Program:
    def Start(self):
        editor = AI4Animation.Scene.AddEntity("MotionEditor")

        editor.AddComponent(
            MotionEditor,
            Dataset(
                npz_path,
                [
                    lambda x: RootModule(
                        x,
                        Definitions.HipName,
                        Definitions.LeftHipName,
                        Definitions.RightHipName,
                        Definitions.LeftShoulderName,
                        Definitions.RightShoulderName,
                    ),
                    lambda x: MotionModule(x),
                    lambda x: ContactModule(
                        x,
                        [
                            (Definitions.LeftAnkleName, 0.1, 0.25),
                            (Definitions.LeftBallName, 0.05, 0.25),
                            (Definitions.RightAnkleName, 0.1, 0.25),
                            (Definitions.RightBallName, 0.05, 0.25),
                        ],
                    ),
                    lambda x: GuidanceModule(x),
                ],
            ),
            model_path,
            bone_names
        )

        AI4Animation.Standalone.Camera.SetTarget(editor)

    def Update(self):
        pass


if __name__ == "__main__":
    AI4Animation(Program())
```

**What this does:**

1. Creates a `Dataset` pointing to a folder of NPZ motion files
2. Attaches animation modules (root trajectory, motion, contacts, guidance)
3. The `MotionEditor` component provides a GUI timeline for scrubbing through clips
4. Camera follows the editor entity

<video controls autoplay loop muted width="100%">
  <source src="../../assets/videos/MotionEditor.mp4" type="video/mp4">
</video>
