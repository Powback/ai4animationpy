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
import os
import sys
from pathlib import Path

from ai4animation import Actor, AI4Animation, Rotation, Time, Vector3

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent / "_ASSETS_/Cranberry")

sys.path.append(ASSETS_PATH)
import Definitions


class Program:
    def Start(self):
        entity = AI4Animation.Scene.AddEntity("Actor")
        model_path = os.path.join(ASSETS_PATH, "Model.glb")
        self.Actor = entity.AddComponent(
            Actor, model_path, Definitions.FULL_BODY_NAMES, True
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
2. Attaches an `Actor` component, loading a GLB model with the specified bone names
3. Rotates the character every frame based on elapsed time
4. Calls `SyncFromScene()` to update internal bone transforms from the scene graph

---

## Playing Motion Data

Use the `MotionEditor` component to browse and play motion clips from NPZ files:

```python
import os
import sys
from pathlib import Path

from ai4animation import (
    AI4Animation,
    ContactModule,
    Dataset,
    MotionEditor,
    MotionModule,
    GuidanceModule,
    RootModule,
)

SCRIPT_DIR = Path(__file__).parent
ASSETS_PATH = str(SCRIPT_DIR.parent / "_ASSETS_/Cranberry")

sys.path.append(ASSETS_PATH)
import Definitions


class Program:
    def Start(self):
        editor = AI4Animation.Scene.AddEntity("MotionEditor")

        editor.AddComponent(
            MotionEditor,
            Dataset(
                os.path.join(ASSETS_PATH, "Motions"),
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
            os.path.join(ASSETS_PATH, "Model.glb"),
            Definitions.FULL_BODY_NAMES,
        )

        AI4Animation.Standalone.Camera.SetTarget(editor)

    def Update(self):
        pass


if __name__ == "__main__":
    AI4Animation(Program())
```

**What this does:**

1. Creates a `Dataset` pointing to a folder of NPZ motion files
2. Attaches animation modules (root trajectory, motion, contacts, guidance) via lambda factories
3. The `MotionEditor` component provides a GUI timeline for scrubbing through clips
4. Camera follows the editor entity

!!! tip
    The default mode is `STANDALONE` — no need to specify it explicitly.
