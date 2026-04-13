# Demo Programs

Reference table of all demo programs included in the repository. Each demo illustrates specific framework concepts and can be run directly.

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
| **[Locomotion/Biped](https://github.com/facebookresearch/ai4animationpy/tree/main/Demos/Locomotion/Biped)** | Stylized biped neural network locomotion | Full inference pipeline, `FeedTensor`/`ReadTensor`, IK | Standalone |
| **[Locomotion/Quadruped](https://github.com/facebookresearch/ai4animationpy/tree/main/Demos/Locomotion/Quadruped)** | Quadruped neural network locomotion | Gait transitions, action poses, `GuidanceModule` | Standalone |
| **[AI/ToyExample](https://github.com/facebookresearch/ai4animationpy/tree/main/Demos/AI/ToyExample)** | Minimal training example | `MLP`, optimizer, scheduler, loss history | Standalone |
| **[AI/Autoencoder](https://github.com/facebookresearch/ai4animationpy/tree/main/Demos/AI/Autoencoder)** | Autoencoder training on motion data | VAE, motion features, `DataSampler` | Standalone |

---

## Video Previews

The following clips are recorded from the included demos and workflows.

### Biped Locomotion

<video controls preload="metadata" width="100%">
	<source src="../../assets/videos/Biped_Locomotion.mp4" type="video/mp4">
	Your browser does not support the video tag.
</video>

### Quadruped Locomotion

<video controls preload="metadata" width="100%">
	<source src="../../assets/videos/Quadruped_Locomotion.mp4" type="video/mp4">
	Your browser does not support the video tag.
</video>

### Training Toy Example

<video controls preload="metadata" width="100%">
	<source src="../../assets/videos/TrainingToyExample.mp4" type="video/mp4">
	Your browser does not support the video tag.
</video>

### Training Future Motion Prediction

<video controls preload="metadata" width="100%">
	<source src="../../assets/videos/Training.mp4" type="video/mp4">
	Your browser does not support the video tag.
</video>

### ECS

<video controls preload="metadata" width="100%">
	<source src="../../assets/videos/ECS.mp4" type="video/mp4">
	Your browser does not support the video tag.
</video>

### Actor

<video controls preload="metadata" width="100%">
	<source src="../../assets/videos/Actor.mp4" type="video/mp4">
	Your browser does not support the video tag.
</video>

### Motion Import (GLB/BVH/NPZ)

<video controls preload="metadata" width="100%">
	<source src="../../assets/videos/MocapImport.mp4" type="video/mp4">
	Your browser does not support the video tag.
</video>

### Motion Editor

<video controls preload="metadata" width="100%">
	<source src="../../assets/videos/MotionEditor.mp4" type="video/mp4">
	Your browser does not support the video tag.
</video>

### Inverse Kinematics

<video controls preload="metadata" width="100%">
	<source src="../../assets/videos/IK.mp4" type="video/mp4">
	Your browser does not support the video tag.
</video>
