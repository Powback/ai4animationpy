# Installation

AI4AnimationPy requires Python 3.12+ and uses Conda for environment management. Select your platform below for the appropriate setup instructions.

---

## Environment Setup

=== "Windows"

    ```bash
    conda create -n AI4AnimationPY python=3.12
    conda activate AI4AnimationPY
    pip install msvc-runtime==14.40.33807
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    pip install nvidia-cudnn-cu12==9.3.0.75 nvidia-cuda-runtime-cu12==12.5.82 nvidia-cufft-cu12==11.2.3.61
    pip install onnxruntime-gpu==1.19.0
    pip install -e . --use-pep517
    ```

=== "Linux"

    ```bash
    conda create -y -n AI4AnimationPY python=3.12 pip
    conda activate AI4AnimationPY
    pip install torch torchvision torchaudio onnx raylib numpy scipy matplotlib scikit-learn einops pygltflib pyscreenrec tqdm pyyaml ipython
    pip install onnxruntime-gpu
    pip install -e . --no-dependencies
    ```

=== "macOS"

    ```bash
    conda create -y -n AI4AnimationPY python=3.12 pip
    conda activate AI4AnimationPY
    pip install torch torchvision torchaudio onnx raylib numpy scipy matplotlib scikit-learn einops pygltflib pyscreenrec tqdm pyyaml ipython
    pip install onnxruntime
    pip install -e . --no-dependencies
    ```
!!! note
    You may need to adjust the PyTorch/CUDA version based on your GPU.

---

## Install in Development Mode

After setting up the Conda environment, install the package in editable (development) mode from the repository root:

```bash
pip install -e .
```

This allows you to modify the source code and have changes reflected immediately without reinstalling.

---

## Verify Installation

Run the minimal "Hello World" demo to verify everything is working:

```bash
python Demos/Empty/Program.py
```

If no errors appear, the framework is installed correctly. For standalone mode (with rendering), run:

```bash
python Demos/Actor/Program.py
```

This should open a window displaying a character model.
