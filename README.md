# robot_lab

piper on Isaac Lab. based on https://github.com/fan-ziqi/robot_lab
This code repository is using IsaacLab-v1.4.1.
## Get Ready

You need to install `Isaac Lab`.

## Installation

Using a python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e ./exts/robot_lab
```

## Try examples

go2-piper

```bash
# Train
python scripts/rsl_rl/train.py --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2-Piper-v0 --num_envs 4096 
# Play
python scripts/rsl_rl/play.py --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2-Piper-v0
```

piper open drawer

```bash
# Train
python scripts/rsl_rl/train.py --task Isaac-Open-Drawer-Piper --num_envs 4096 
# Play
python scripts/rsl_rl/play.py --task Isaac-Open-Drawer-Piper
```

piper lift cube

```bash
# Train
python scripts/rsl_rl/train.py --task Isaac-Lift-Cube-Piper --num_envs 4096 
# Play
python scripts/rsl_rl/play.py --task Isaac-Lift-Cube-Piper
```

piper reach point

```bash
# Train
python scripts/rsl_rl/train.py --task Isaac-Reach-Piper --num_envs 4096 
# Play
python scripts/rsl_rl/play.py --task Isaac-Reach-Piper
```

The above configs are flat, you can change Flat to Rough

**Note**

* Record video of a trained agent (requires installing `ffmpeg`), add `--video --video_length 200`
* Play/Train with 32 environments, add `--num_envs 32`
* Play on specific folder or checkpoint, add `--load_run run_folder_name --checkpoint model.pt`
* Resume training from folder or checkpoint, add `--resume --load_run run_folder_name --checkpoint model.pt`

Check [import_new_asset](https://docs.robotsfan.com/isaaclab/source/how-to/import_new_asset.html) for detail

## Tensorboard

To view tensorboard, run:

```bash
tensorboard --logdir=logs
```

## Code formatting

A pre-commit template is given to automatically format the code.

To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```
