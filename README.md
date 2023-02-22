> Repo under construction- please check back soon! 

# Embodied Scene-aware Human Pose Estimation

[[paper]](https://arxiv.org/abs/2206.09106) [[website]](https://zhengyiluo.github.io/projects/embodied_pose/) [[Video]](https://www.youtube.com/watch?v=8Ae0xzqAtm8)

<div float="center">
  <img src="assets/gif/teaser.gif" />
</div>

## Dependencies

To create the environment, follow the following instructions: 

1. Create new conda environment and install pytroch:
```
conda create -n embodiedpose python=3.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

2. Download and setup mujoco: [Mujoco](http://www.mujoco.org/)
```
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco
mv mujoco210 ~/.mujoco/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
```

3. Download and install Universal Humanoid Controller Locally and follow the instructions to setup the data. Make sure you have UHC running locally before proceeding:

```
git clone git@github.com:ZhengyiLuo/UniversalHumanoidControl.git 
cd UniversalHumanoidControl
pip install -e .
```

