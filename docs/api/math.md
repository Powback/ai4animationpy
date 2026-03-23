# Math Library

A dual-backend math library providing vectorized operations for animation. All functions accept batched tensors of arbitrary leading dimensions.

**Files:** `ai4animation/Math/Tensor.py`, `Transform.py`, `Rotation.py`, `Quaternion.py`, `Vector3.py`

---

## Tensor

The foundational dual-backend layer wrapping NumPy and PyTorch with a unified API.

### Backend Switching

```python
from ai4animation.Math.Tensor import Tensor, Backend

# Default: NumPy
Tensor.DefaultBackend = Backend.NumPy

# Switch to PyTorch (GPU-accelerated)
Tensor.DefaultBackend = Backend.PyTorch

# Per-operation override
result = Tensor.Create([1, 2, 3], backend=Backend.PyTorch)
```

When using the PyTorch backend, tensors are automatically placed on GPU if CUDA is available.

### Key Operations

| Operation | Description |
|-----------|-------------|
| `Tensor.Create(data)` | Create tensor from list/array |
| `Tensor.Eye(n)` | Identity matrix |
| `Tensor.Zeros(shape)` | Zero tensor |
| `Tensor.Ones(shape)` | Ones tensor |
| `Tensor.Normalize(x)` | Unit normalize |
| `Tensor.MatMul(a, b)` | Matrix multiplication |
| `Tensor.Inverse(x)` | Matrix inverse |
| `Tensor.Gaussian(x, sigma)` | Gaussian smoothing |
| `Tensor.Interpolate(a, b, t)` | Linear interpolation |
| `Tensor.Sin(x)` / `Cos(x)` | Trigonometric functions |
| `Tensor.Abs(x)` | Absolute value |
| `Tensor.ToDevice(x)` | Move to GPU if available |

---

## Transform

4×4 homogeneous transformation matrices for combined translation and rotation.

### Representation

```
[R R R Tx]
[R R R Ty]
[R R R Tz]
[0 0 0  1]
```

Where the upper-left 3×3 is a rotation matrix and the right column stores translation.

### Key Operations

| Operation | Description |
|-----------|-------------|
| `Transform.Identity()` | 4×4 identity matrix |
| `Transform.TR(position, rotation)` | Translation + Rotation matrix |
| `Transform.TRS(position, rotation, scale)` | Translation + Rotation + Scale |
| `Transform.GetPosition(t)` | Extract position `[..., 3]` |
| `Transform.GetRotation(t)` | Extract rotation `[..., 3, 3]` |
| `Transform.SetPosition(t, pos)` | Set position component |
| `Transform.SetRotation(t, rot)` | Set rotation component |
| `Transform.Multiply(a, b)` | Compose transforms |
| `Transform.Inverse(t)` | Invert transform |
| `Transform.TransformationTo(a, b)` | Relative transform from a to b |
| `Transform.TransformationFrom(a, b)` | Apply relative transform |
| `Transform.GetMirror(t, axis)` | Mirror across axis |

---

## Rotation

3×3 rotation matrices.

### Key Operations

| Operation | Description |
|-----------|-------------|
| `Rotation.Identity()` | 3×3 identity |
| `Rotation.Euler(x, y, z)` | From Euler angles (degrees) |
| `Rotation.RotationX(angle)` | Rotation about X axis |
| `Rotation.RotationY(angle)` | Rotation about Y axis |
| `Rotation.RotationZ(angle)` | Rotation about Z axis |
| `Rotation.Look(forward, up)` | Look-at rotation matrix |
| `Rotation.LookPlanar(forward)` | Planar look-at (Y-up) |
| `Rotation.Multiply(a, b)` | Compose rotations |
| `Rotation.MultiplyVector(r, v)` | Rotate a vector |
| `Rotation.Normalize(r)` | Orthonormalize rotation |
| `Rotation.Inverse(r)` | Transpose (inverse) |

---

## Quaternion

4-element quaternions in `[x, y, z, w]` format.

### Key Operations

| Operation | Description |
|-----------|-------------|
| `Quaternion.Identity()` | `[0, 0, 0, 1]` |
| `Quaternion.Euler(x, y, z)` | From Euler angles (degrees) |
| `Quaternion.AngleAxis(angle, axis)` | From angle-axis |
| `Quaternion.FromTo(from_dir, to_dir)` | Rotation between two directions |
| `Quaternion.ToMatrix(q)` | Convert to 3×3 rotation matrix |
| `Quaternion.FromMatrix(m)` | Convert from 3×3 rotation matrix |
| `Quaternion.Multiply(a, b)` | Compose quaternions |
| `Quaternion.Conjugate(q)` | Conjugate (inverse for unit quaternions) |
| `Quaternion.Inverse(q)` | Full inverse |
| `Quaternion.Slerp(a, b, t)` | Spherical interpolation |

---

## Vector3

3-element vectors for positions, directions, and velocities.

### Key Operations

| Operation | Description |
|-----------|-------------|
| `Vector3.Create(x, y, z)` | Create vector |
| `Vector3.Zero()` | `[0, 0, 0]` |
| `Vector3.One()` | `[1, 1, 1]` |
| `Vector3.Up()` | `[0, 1, 0]` |
| `Vector3.Forward()` | `[0, 0, 1]` |
| `Vector3.Normalize(v)` | Unit normalize |
| `Vector3.Cross(a, b)` | Cross product |
| `Vector3.Dot(a, b)` | Dot product |
| `Vector3.Distance(a, b)` | Euclidean distance |
| `Vector3.Lerp(a, b, t)` | Linear interpolation |
| `Vector3.Slerp(a, b, t)` | Spherical interpolation |
| `Vector3.SignedAngle(a, b, axis)` | Signed angle between vectors |
| `Vector3.PositionTo(pos, transform)` | Transform position into local space |
| `Vector3.PositionFrom(pos, transform)` | Transform position from local space |
| `Vector3.DirectionTo(dir, transform)` | Transform direction into local space |
| `Vector3.DirectionFrom(dir, transform)` | Transform direction from local space |

---

## Batching

All math operations support **arbitrary leading batch dimensions**:

```python
# Single transform
t = Transform.Identity()  # shape [4, 4]

# Batch of 10 transforms
t = Transform.Identity(batch=(10,))  # shape [10, 4, 4]

# Batch of [32, 24] transforms (e.g., 32 frames, 24 joints)
positions = motion.GetBonePositions(timestamps, bones)  # [32, 24, 3]
```

This enables efficient vectorized computation across frames, joints, and batch dimensions without explicit loops.
