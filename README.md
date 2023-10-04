
# Embodied Scene-aware Human Pose Estimation

Official implementation of NeurIPS 2022 paper: "Embodied Scene-aware Human Pose Estimation". In this paper, we estimate 3D poses based on a simulated agent's proprioception and scene awareness, along with external third-person observations.

[[paper]](https://arxiv.org/abs/2206.09106) [[website]](https://zhengyiluo.github.io/projects/embodied_pose/) [[Video]](https://www.youtube.com/watch?v=8Ae0xzqAtm8)

<div float="center">
    <img src="assets/gif/wild_demo1.gif" />
    <img src="assets/gif/wild_demo2.gif" />
  <img src="assets/gif/teaser.gif" />
</div>

## News 🚩


[June 18, 2023 ] Releaseing in-the-wild Demo. Please note that this is mainly as a proof-of-concept as EmbodiedPose is great at recovering global motion, but isn't so good at capturing high-frequecy movement

[March 31, 2023 ] Training code released.

[February 25, 2023 ] Evaluation code released.


## Introduction
In this project, we develop "Embodied Human Pose Estimation", where we control a humanoid robot to move in a scene and estimate the human pose. Using 2D keypoint and scene information (camera pose and scene) as input, we estimate global pose in a casual fashion.

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

3. Download and install [Universal Humanoid Controller]([url](https://github.com/ZhengyiLuo/UniversalHumanoidControl)) locally and **follow the instructions to setup the data and download the models**. **❗️❗️❗️Make sure you have UHC running locally before proceeding**:

```
git clone git@github.com:ZhengyiLuo/UniversalHumanoidControl.git 
cd UniversalHumanoidControl
pip install -e .
bash download_data.sh
```


## Data processing for evaluating/training UHC

EmbodiedPose is trained on a combinatino of AMASS, kinpoly, and h36m motion dataset. We generate paired 2D keypoints from the motion captre data and randomly selected camera information. 
Use the following script to download trained models, evaluation data, and pretrained [humor models](https://github.com/davrempe/humor/blob/main/get_ckpt.sh).

```
bash download_data.sh
```

Download SMPL paramters from [SMPL](https://smpl.is.tue.mpg.de/). Put them in the `data/smpl` folder, unzip them into 'data/smpl' folder. Please download the v1.1.0 version, which contains the neutral humanoid. Rename the files `basicmodel_neutral_lbs_10_207_0_v1.1.0`, `basicmodel_m_lbs_10_207_0_v1.1.0.pkl`, `basicmodel_f_lbs_10_207_0_v1.1.0.pkl` to `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl` and `SMPL_FEMALE.pkl`. The file structure should look like this:

```

|-- data
    |-- smpl
        |-- SMPL_FEMALE.pkl
        |-- SMPL_NEUTRAL.pkl
        |-- SMPL_MALE.pkl
        
```

If you wish to train EmbodiedPose, first, download the AMASS dataset from [AMASS](https://amass.is.tue.mpg.de/). Then, run the following script on the unzipped data:
 

```
python uhc/data_process/process_amass_raw.py
```

which dumps the data into the `amass_db_smplh.pt` file. Then, run 

```
python uhc/data_process/process_amass_db.py
```

For processing your own SMPL data for training, you can refer to the required data fields in `process_amass_db.py`. 


## Evaluation 
```
python scripts/eval_scene.py --cfg tcn_voxel_4_5 --epoch -1
```

## Evaluate on In-the-wild Data 

To run EmbodiedPose on in-the-wild data (mainly as a proof-of-concept, as EmbodiedPose is really great at recovering global motion, but isn't great at capturing high-frequecy movement), we will use [HybrIK](https://github.com/Jeff-sjtu/HybrIK) to generate the camera information, 2D keypoints, initialization pose. We do not use the 3D pose estimation (though directly use UHC to imitate the pose is possible) and only uses the 2D keypoints. 

First, run HybrIK on the in-the-wild following their instructions:

```
python scripts/demo_video.py --video-name assets/demo_videos/embodied_demo.mp4 --out-dir sample_data/res_dance --save-pk 
```

Using the saved pk file, we will further process it into the format that EmbodiedPose can use using the script `process_hybrik_data.py`. Details of how to debug this script can be found in the notebook `in_the_wild_poc.ipynb`. 


```
python scripts/process_hybrik_data.py --input sample_data/res_dance/res.pk --output sample_data/wild_processed.pkl
```

Finally, run the following script to evaluate the model on the in-the-wild data. 
```
python scripts/eval_scene.py --cfg tcn_voxel_4_5 --epoch -1 --data sample_data/wild_processed.pkl
```



## Training

```
python scripts/train_models.py --cfg tcn_voxel_4_5 
```

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



