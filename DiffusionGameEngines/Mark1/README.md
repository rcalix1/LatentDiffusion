## Starting GameNgen ideas

* MNIST
* Stable diffusion
* 


![GameNGen demo](anim.gif)


# GameNGen – Generative Environment Modeling for RL Agents

**GameNGen** is a research and prototyping framework for learning visual world models that simulate environment transitions. The goal is to generate the next frame of a game or simulation given the previous frame(s) and control input, and use this model to train reinforcement learning (RL) agents in both real and generated environments.

---

## ✨ Project Summary

This project aims to:

* Train a conditional generative model (initially a Stable Diffusion variant) to predict visual frames conditioned on control input and previous state.
* Replace simple datasets (MNIST) with image-based game environments like CartPole, LunarLander, and VizDoom.
* Train RL agents (PPO or DQN) on real environments.
* Later, use the GameNGen model to simulate rollouts and train agents in "dream" environments.

---

## 📚 Environment Setup (Python 3.10 Compatible)

Install the required packages:

```bash
pip install torch torchvision matplotlib imageio
pip install ipywidgets
pip install stable-baselines3[extra]
pip install gym
pip install vizdoom
```

If you're using Jupyter:

```bash
pip install notebook
jupyter nbextension enable --py widgetsnbextension
```

Optional (for symbolic environments like MiniGrid):

```bash
pip install gym-minigrid
```

---

## 🌐 Supported Environments

### 1. MNIST (Prototype Phase)

* Uses digit labels as control vectors.
* Good for bootstrapping the diffusion model.
* Output: simple grayscale images.

### 2. CartPole-v1 (Gym Classic Control)

* 4D state vector: position, velocity, angle, angular velocity.
* Actions: `left`, `right`.
* `env.render(mode="rgb_array")` provides images for GameNGen.

### 3. LunarLander-v2

* Discrete 8-action space: fire side/main thrusters.
* 2D simulation with gravity and descent control.
* Great for future transition to plane control.

### 4. MountainCar-v0

* Discrete actions: `left`, `right`, `no push`.
* Encourages planning under low reward.

### 5. VizDoom

* First-person 3D environment.
* Discrete actions: `move_forward`, `turn_left`, `shoot`, etc.
* Excellent for visual world modeling and rich RL tasks.

### 6. MiniGrid (Optional)

* Symbolic 2D environment for goal-driven agents.
* Low-dimensional inputs.

### 7. X-Plane (Future Phase)

* Realistic 6-DoF control: pitch, yaw, roll.
* Will be used once PPO and GameNGen are proven.

---

## 📊 Project Roadmap

| Phase      | Objective                                                           |
| ---------- | ------------------------------------------------------------------- |
| ✅ Phase 1  | Train Stable Diffusion on MNIST with 10D control vector             |
| ⚒ Phase 2  | Add previous image + label conditioning                             |
| ⚒ Phase 3  | Switch to Gym/VizDoom and collect (image, action, next\_image) data |
| ⚒ Phase 4  | Train PPO agent on real environment                                 |
| ⚒ Phase 5  | Train GameNGen to simulate transitions                              |
| 🧠 Phase 6 | Train agent in generated dream environments                         |

---

## 📁 Recommended File Structure

```
GameNGen/
├── models/               # Diffusion + PPO model definitions
├── data/                 # Collected trajectories and image pairs
├── notebooks/            # Experiment notebooks (MNIST, CartPole, VizDoom)
├── utils/                # Helper functions: render, GIF generation, logging
├── gifs/                 # Saved rollout animations
└── README.md
```

---

## 🚷 Next Steps

* [ ] Add temporal conditioning to diffusion model.
* [ ] Generate rollout GIFs and embed in GitHub.
* [ ] Implement `collect_data.py` for CartPole and VizDoom.
* [ ] Train PPO baseline agent using `stable-baselines3`.
* [ ] Train GameNGen on collected transition data.
* [ ] Migrate to dream-based agent training.

---

## 🎓 Notes

* All environments and tools are tested with **Python 3.10**.
* VizDoom is the ideal next environment: visually rich but easy to install.
* X-Plane will be integrated at a later phase when high-fidelity flight is needed.

---

## ⚙ Optional Enhancements

* Convert animations to `.gif` for GitHub compatibility
* Add Jupyter-compatible HTML visualizations using `to_jshtml()`
* Embed rollout predictions and ground truth comparisons
* Include reward overlays or action labels on generated frames

---

