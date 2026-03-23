# Importers

Import 3D character models and animations from industry-standard formats (GLB, FBX, BVH) via a common abstract interface.

**Files:** `ai4animation/Import/ModelImporter.py`, `GLBImporter.py`, `FBXImporter.py`, `BVHImporter.py`, `BatchConverter.py`

---

## ModelImporter (Abstract Base)

**File:** `ai4animation/Import/ModelImporter.py`

All importers implement this interface:

| Property | Type | Description |
|----------|------|-------------|
| `JointNames` | `List[str]` | Ordered joint name list |
| `JointParents` | `List[str]` | Parent name per joint |
| `JointMatrices` | `ndarray [J, 4, 4]` | Bind-pose joint matrices |
| `Meshes` | `List[Mesh]` | Mesh geometry data |
| `Skin` | `Optional[Skin]` | Skinning data |

**Method:**

| Method | Description |
|--------|-------------|
| `LoadMotion(names, floor)` | Returns a `Motion` object from animation data |

---

## GLBImporter

**File:** `ai4animation/Import/GLBImporter.py`

Parses `.glb` (glTF Binary) files using the `pygltflib` library. Extracts nodes, skins, meshes, and animations.

```python
from ai4animation.Import.GLBImporter import GLB

model = GLB.Create("character.glb")
joint_names = model.JointNames
motion = model.LoadMotion()
```

---

## FBXImporter

**File:** `ai4animation/Import/FBXImporter.py`

Custom parser for `.fbx` (Autodesk FBX) files. Extracts joints, meshes, skin data, and animations.

```python
from ai4animation.Import.FBXImporter import FBX

model = FBX.Create("character.fbx")
motion = model.LoadMotion()
```

---

## BVHImporter

**File:** `ai4animation/Import/BVHImporter.py`

Parses `.bvh` (BioVision Hierarchy) motion capture files. Extracts skeleton hierarchy and motion channel data.

```python
from ai4animation.Import.BVHImporter import BVH

model = BVH.Create("motion.bvh")
motion = model.LoadMotion()
```

---

## Data Classes

### Mesh

```python
class Mesh:
    Name: str
    Vertices: ndarray   # [V, 3]
    Normals: ndarray    # [V, 3]
    Triangles: ndarray  # [T*3]
    SkinIndices: ndarray # [V, K] — bone indices per vertex
    SkinWeights: ndarray # [V, K] — blend weights per vertex
    HasSkinning: bool
```

### Skin

```python
class Skin:
    Inverse_bind_matrices: ndarray  # [J, 4, 4]
    Joints: ndarray                  # [J] — joint indices
```

---

## BatchConverter (CLI)

**File:** `ai4animation/Import/BatchConverter.py`

Command-line tool for batch converting GLB, FBX, and BVH files to the internal NPZ format.

```bash
convert -h
```

This discovers all supported files in a directory and converts them to NPZ, ready for use with `Dataset`.

---

## Importer Comparison

| Importer | Format | Library | Mesh | Skin | Animation |
|----------|--------|---------|------|------|-----------|
| **GLBImporter** | `.glb` | pygltflib | ✅ | ✅ | ✅ |
| **FBXImporter** | `.fbx` | Custom parser | ✅ | ✅ | ✅ |
| **BVHImporter** | `.bvh` | Custom parser | ❌ | ❌ | ✅ |
