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
