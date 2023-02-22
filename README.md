# Embodied Scene-aware Human Pose Estimation

> Repo under construction.  

TODOS:

- [x] Code runnable
- [ ] Training code 


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
pip install -r requirements.txt
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

## Download data
```
bash download_data.sh
```

## Evaluation 
```
python scripts/eval_scene.py --cfg tcn_voxel_4_5 --epoch -1
```

## Training
TBD

## Citation
If you find this work useful for your research, please cite our paper:
```
@inproceedings{Luo2022EmbodiedSH,
  title={Embodied Scene-aware Human Pose Estimation},
  author={Zhengyi Luo and Shun Iwase and Ye Yuan and Kris Kitani},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## References
This repository is built on top of the following amazing repositories:
* State transition code is from: [HuMoR](https://github.com/davrempe/humor)
* Part of the UHC code is from: [rfc](https://github.com/Khrylx/RFC)
* SMPL models and layer is from: [SMPL-X model](https://github.com/vchoutas/smplx)
* Feature extractors are from: [SPIN](https://github.com/nkolot/SPIN)
* NN modules are from (khrylib): [DLOW](https://github.com/Khrylx/DLow)



