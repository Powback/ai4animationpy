# Tutorial: Adding an Importer

This tutorial shows how to implement a custom `ModelImporter` for a new 3D file format.

---

## Step 1: Understand the Interface

All importers implement the `ModelImporter` abstract base class:

```python
from ai4animation.Import.ModelImporter import ModelImporter, Mesh, Skin
```

### Required Properties

| Property | Return Type | Description |
|----------|-------------|-------------|
| `JointNames` | `List[str]` | Ordered list of joint names |
| `JointParents` | `List[str]` | Parent name per joint (`None` for root) |
| `JointMatrices` | `ndarray [J, 4, 4]` | Bind-pose joint matrices |
| `Meshes` | `List[Mesh]` | Mesh geometry list |
| `Skin` | `Optional[Skin]` | Skinning data (or `None`) |

### Required Method

| Method | Return Type | Description |
|--------|-------------|-------------|
| `LoadMotion(names, floor)` | `Motion` | Parse animation data into a `Motion` object |

---

## Step 2: Implement the Importer

```python
import numpy as np
from ai4animation.Import.ModelImporter import ModelImporter, Mesh, Skin
from ai4animation.Animation.Motion import Motion, Hierarchy
from ai4animation.Math import Transform, Quaternion


class MyFormatImporter(ModelImporter):
    def __init__(self, filepath):
        self._filepath = filepath
        self._data = self._parse(filepath)

    @staticmethod
    def Create(filepath):
        return MyFormatImporter(filepath)

    def _parse(self, filepath):
        # Parse the file format here
        # Return internal data structure
        pass

    @property
    def JointNames(self):
        return self._data.joint_names

    @property
    def JointParents(self):
        return self._data.joint_parents

    @property
    def JointMatrices(self):
        # Return [J, 4, 4] bind-pose matrices
        return self._data.bind_matrices

    @property
    def Meshes(self):
        meshes = []
        for m in self._data.meshes:
            mesh = Mesh()
            mesh.Name = m.name
            mesh.Vertices = m.vertices      # [V, 3]
            mesh.Normals = m.normals        # [V, 3]
            mesh.Triangles = m.triangles    # [T*3]
            mesh.SkinIndices = m.skin_idx   # [V, K]
            mesh.SkinWeights = m.skin_wgt   # [V, K]
            mesh.HasSkinning = True
            meshes.append(mesh)
        return meshes

    @property
    def Skin(self):
        skin = Skin()
        skin.Inverse_bind_matrices = self._data.ibm  # [J, 4, 4]
        skin.Joints = self._data.joint_indices        # [J]
        return skin

    def LoadMotion(self, names=None, floor=None):
        # Parse animation data into per-frame transforms
        num_frames = self._data.num_frames
        num_joints = len(self.JointNames)
        framerate = self._data.fps

        # Build [F, J, 4, 4] global transforms
        frames = np.zeros((num_frames, num_joints, 4, 4))
        for f in range(num_frames):
            for j in range(num_joints):
                frames[f, j] = self._data.get_transform(f, j)

        hierarchy = Hierarchy(self.JointNames, self.JointParents)
        name = self._data.animation_name

        return Motion(name, hierarchy, frames, framerate)
```

---

## Step 3: Data Classes Reference

### Mesh

```python
class Mesh:
    Name: str              # Mesh name
    Vertices: ndarray      # [V, 3] vertex positions
    Normals: ndarray       # [V, 3] vertex normals
    Triangles: ndarray     # [T*3] triangle indices
    SkinIndices: ndarray   # [V, K] bone indices per vertex
    SkinWeights: ndarray   # [V, K] blend weights per vertex
    HasSkinning: bool      # Whether skinning data exists
```

### Skin

```python
class Skin:
    Inverse_bind_matrices: ndarray  # [J, 4, 4] inverse bind-pose
    Joints: ndarray                  # [J] joint indices
```

---

## Step 4: Use Your Importer

```python
from ai4animation import Actor, AI4Animation

class Program:
    def Start(self):
        entity = AI4Animation.Scene.AddEntity("Character")
        # Actor will use your importer if registered
        actor = entity.AddComponent(Actor, "model.myformat", bone_names)
```

Or load motion directly:

```python
model = MyFormatImporter.Create("model.myformat")
motion = model.LoadMotion()
motion.SaveToNPZ("output.npz")
```

---

## Existing Importers as Reference

Study the built-in importers for implementation patterns:

| Importer | File | Complexity |
|----------|------|-----------|
| `BVHImporter` | `ai4animation/Import/BVHImporter.py` | Simplest — text-based, hierarchy + channels |
| `GLBImporter` | `ai4animation/Import/GLBImporter.py` | Medium — binary format via pygltflib |
| `FBXImporter` | `ai4animation/Import/FBXImporter.py` | Most complex — custom binary parser |

!!! tip
    Start with the `BVHImporter` as a reference — it's the simplest implementation and clearly shows the required interface.
