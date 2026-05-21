# Environment Management

This project uses **Conda** and **uv** together.

Conda is the source of truth for the Python environment and dependencies, especially native/scientific packages such as NumPy, SciPy, scikit-learn, LightGBM, PyArrow, and PyTorch. This avoids macOS OpenMP conflicts between LightGBM and scikit-learn.

uv is used only as a fast command runner inside the active Conda environment.

## Why both Conda and uv?
Conda provides a stable native package stack for ML dependencies. This is important because pip wheels can load conflicting OpenMP runtimes on macOS, especially with LightGBM and scikit-learn.
uv provides a fast and consistent way to run project commands without replacing Conda as the environment manager.