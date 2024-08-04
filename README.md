<!-- Improved compatibility of back to top link: See: https://github.com/ZHU-Zhiyu/NVS_Solver/pull/73 -->
<a name="readme-top"></a>

[![Issues][issues-shield]][issues-url]
<!-- [![MIT License][license-shield]][license-url] -->
<!-- [![MyHomePage][linkedin-shield]][linkedin-url] -->

<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">NVS-Solver: Video Diffusion Model as Zero-Shot Novel View Synthesizer</h3>
  <p align="center">
    <a href="https://arxiv.org/abs/2405.15364">Arxiv</a>
    ·
    <a href="https://github.com/ZHU-Zhiyu/NVS_Solver/issues">Report Bug</a>
    ·
    <a href="https://github.com/ZHU-Zhiyu/NVS_Solver/issues">Request Feature</a>
  </p>
</div>


| Single-View       | Ignatius                                | Family                                | Palace                                | Church                                | Barn                                |
|----------------|----------------|----------------|----------------|----------------|----------------|
| Text2Nerf | <img src='./Assets/Single/text2nerf_gif/ignatius.gif' width='200'> | <img src='./Assets/Single/text2nerf_gif/family.gif' width='200'> | <img src='./Assets/Single/text2nerf_gif/palace.gif' width='200'> | <img src='./Assets/Single/text2nerf_gif/church.gif' width='200'> | <img src='./Assets/Single/text2nerf_gif/barn.gif' width='200'> |
| MotionCtrl| <img src='./Assets/Single/motionctrl_gif/ignatius.gif' width='200'> | <img src='./Assets/Single/motionctrl_gif/family.gif' width='200'> | <img src='./Assets/Single/motionctrl_gif/palace.gif' width='200'> | <img src='./Assets/Single/motionctrl_gif/church.gif' width='200'> | <img src='./Assets/Single/motionctrl_gif/barn.gif' width='200'> |
| NVS-Solver (DGS)  | <img src='./Assets/Single/Ours_DGS_gif/ignatius.gif' width='200'> | <img src='./Assets/Single/Ours_DGS_gif/family.gif' width='200'> | <img src='./Assets/Single/Ours_DGS_gif/palace.gif' width='200'> | <img src='./Assets/Single/Ours_DGS_gif/church.gif' width='200'> | <img src='./Assets/Single/Ours_DGS_gif/barn.gif' width='200'> |
| NVS-Solver (Post) | <img src='./Assets/Single/Ours_gif/ignatius.gif' width='200'> | <img src='./Assets/Single/Ours_gif/family.gif' width='200'> | <img src='./Assets/Single/Ours_gif/palace.gif' width='200'> | <img src='./Assets/Single/Ours_gif/church.gif' width='200'> | <img src='./Assets/Single/Ours_gif/barn.gif' width='200'> |


| Multi-View       |  Truck                              | Playground | Caterpillar                                | Scan55                                | Scan3                                |
|----------------|----------------|----------------|----------------|----------------|-----------------|
| 3D-Aware | <img src='./Assets/multi/ivid_multi_01/truck.gif' width='200'> | <img src='./Assets/multi/ivid_multi_01/3daware_playground.gif' width='200'> | <img src='./Assets/multi/ivid_multi_01/caterpillar.gif' width='200'> | <img src='./Assets/multi/ivid_multi_01/scan55.gif' width='200'> | <img src='./Assets/multi/ivid_multi_01/scan3.gif' width='200'> |
| MotionCtrl| <img src='./Assets/multi/mctrl_01/truck_R1.gif' width='200'> | <img src='./Assets/multi/mctrl_01/Mctrl_playground_R1.gif' width='200'> | <img src='./Assets/multi/mctrl_01/caterpillar_R1.gif' width='200'> | <img src='./Assets/multi/mctrl_01/scan55_R1.gif' width='200'> | <img src='./Assets/multi/mctrl_01/scan3_R1.gif' width='200'> |
| NVS-Solver (DGS)  | <img src='./Assets/multi/multi_ours_dgs_01/truck_generated.gif' width='200'> | <img src='./Assets/multi/multi_ours_dgs_01/DGS_playground_generated.gif' width='200'>  | <img src='./Assets/multi/multi_ours_dgs_01/caterpillar_generated.gif' width='200'> | <img src='./Assets/multi/multi_ours_dgs_01/scan55_generated.gif' width='200'> | <img src='./Assets/multi/multi_ours_dgs_01/scan3_generated.gif' width='200'> |
| NVS-Solver (Post) | <img src='./Assets/multi/ours_01/truck_generated.gif' width='200'> | <img src='./Assets/multi/ours_01/Ours_playground_generated.gif' width='200'> | <img src='./Assets/multi/ours_01/caterpillar_generated.gif' width='200'> | <img src='./Assets/multi/ours_01/scan55_generated.gif' width='200'> | <img src='./Assets/multi/ours_01/scan3_generated.gif' width='200'> |


| Mono-Vid       |  Train                              | Bus | Kangroo                                | Train                                | Deer                                | Street                                |
|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| 4DGS | <img src='./Assets/dynamic/4dgs_gif/train5_2.gif' width='200'> | <img src='./Assets/dynamic/4dgs_gif/Gaussian_bus_1.gif' width='200'> | <img src='./Assets/dynamic/4dgs_gif/kangroo2_1.gif' width='200'> | <img src='./Assets/dynamic/4dgs_gif/train3.gif' width='200'> | <img src='./Assets/dynamic/4dgs_gif/Saved_fig.jpg' width='200'> | <img src='./Assets/dynamic/4dgs_gif/street_1.gif' width='200'> |
| MotionCtrl| <img src='./Assets/dynamic/MotionCtrl_gif_01/Motion_ctrltrain5_R2.gif' width='200'> | <img src='./Assets/dynamic/MotionCtrl_gif_01/Motion_ctrl_bus_R1.gif' width='200'> | <img src='./Assets/dynamic/MotionCtrl_gif_01/kangroo_R1.gif' width='200'> | <img src='./Assets/dynamic/MotionCtrl_gif_01/Motion_ctrl_train3_R1.gif' width='200'> | <img src='./Assets/dynamic/MotionCtrl_gif_01/deer_R1.gif' width='200'> | <img src='./Assets/dynamic/MotionCtrl_gif_01/street_R1.gif' width='200'> |
| NVS-Solver (DGS)  | <img src='./Assets/dynamic/Ours_DGS_gif_01/DGS_train52.gif' width='200'> | <img src='./Assets/dynamic/Ours_DGS_gif_01/dgs_bus1.gif' width='200'>  | <img src='./Assets/dynamic/Ours_DGS_gif_01/kanroo1.gif' width='200'> | <img src='./Assets/dynamic/Ours_DGS_gif_01/DGS_train31.gif' width='200'> | <img src='./Assets/dynamic/Ours_DGS_gif_01/deer1.gif' width='200'> | <img src='./Assets/dynamic/Ours_DGS_gif_01/street1.gif' width='200'> |
| NVS-Solver (Post) | <img src='./Assets/dynamic/Ours_gif_01/Ours_train52.gif' width='200'> | <img src='./Assets/dynamic/Ours_gif_01/Ours_bus1.gif' width='200'> | <img src='./Assets/dynamic/Ours_gif_01/kangroo1.gif' width='200'> | <img src='./Assets/dynamic/Ours_gif_01/Ours_train31.gif' width='200'> | <img src='./Assets/dynamic/Ours_gif_01/deer1.gif' width='200'> | <img src='./Assets/dynamic/Ours_gif_01/street1.gif' width='200'> |

[contributors-shield]: https://img.shields.io/github/contributors/ZHU-Zhiyu/NVS_Solver.svg?style=for-the-badge
[contributors-url]: https://github.com/ZHU-Zhiyu/NVS_Solver/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/ZHU-Zhiyu/NVS_Solver.svg?style=for-the-badge
[forks-url]: https://github.com/ZHU-Zhiyu/NVS_Solver/network/members
[stars-shield]: https://img.shields.io/github/stars/ZHU-Zhiyu/NVS_Solver.svg?style=for-the-badge
[stars-url]: https://github.com/ZHU-Zhiyu/NVS_Solver/stargazers
[issues-shield]: https://img.shields.io/github/issues/ZHU-Zhiyu/NVS_Solver.svg?style=for-the-badge
[issues-url]: https://github.com/ZHU-Zhiyu/NVS_Solver/issues
[license-shield]: https://img.shields.io/github/license/ZHU-Zhiyu/NVS_Solver.svg?style=for-the-badge
[license-url]: https://github.com/ZHU-Zhiyu/NVS_Solver/blob/master/LICENSE.txt

## To do list

- [x] Release NVS-Solver for SVD;
- [x] Release NVS-Solver for abitrary trajectory; 
- [ ] Support T2V diffusion models, we are testing some effective [video diffusion models](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard), e.g., [LaVie](https://github.com/Vchitect/LaVie), [t2v-turbo](https://github.com/Ji4chenLi/t2v-turbo);
- [ ] Acceleration with ODE-Solvers, e.g., DPM-Solver;
- [ ] ....

## Getting Started 

### Dependencies

* Linux
* Anaconda 3
* Python 3.9
* CUDA 12.0
* RTX A6000

### Installing

To get started, please create the virtual environment by:
```
python -m venv .env
source .env/bin/activate
```
Please install the PyTorch by:
```
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
```
We use PyTorch 2.2.1 with CUDA 12.0, please install the version corresponding to your CUDA.

Please install the [diffusers](https://huggingface.co/docs/diffusers/index) by:
```
pip install diffusers["torch"] transformers
pip install accelerate
pip install -e ".[torch]"
```
Please install required packages by:
```
pip install -r requirements.txt
```
### Inference

Run
```
bash demo.sh
```
We provide the prompt images utilized in our experiments at [single image](https://drive.google.com/drive/folders/1eEB4U65RlEbqo4s4vTQSBH16929sGvQO?usp=sharing) and [dynamic videos](https://drive.google.com/drive/folders/1w6tyGRg6SIFPGCPBgUffPruWsm7z3pos).
### Our pipeline for abitrary trajectory
<img src='./Assets/arbitrary/1.gif' width='230'> <img src='./Assets/arbitrary/2.gif' width='230'> 

<img src='./Assets/arbitrary/3.gif' width='230'> <img src='./Assets/arbitrary/4.gif' width='230'> 
#### Prepare depth model
First please prepare [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2). As our pipeline read depth as npy file, so please edit the [run.py](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/run.py) for saving predicted depth maps as npy:
1. add this after [line 57](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/run.py#L57)
```
depth_np = depth
```
2. add this after [line 73](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/run.py#L73)
```
np.save(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.npy'),depth_np)
```
#### Prepare your image 
Put your image to a directory, e.g. `/path_to_img`.
```
DIR_PATH=/path_to_img
IMG_PATH=/path_to_img/001.jpg
DEPTH_PATH="${DIR_PATH}/depth"
mkdir -p "$DEPTH_PATH"
```

Predict depth for your image
```
cd /your_path_to_Depth-Anything-V2
python run.py --encoder vitl --img-path "$IMG_PATH" --outdir "$DEPTH_PATH"
```
#### Run the pipeline
```
cd /your_path_to_NVS_Solver
python svd_interpolate_single_img_traj.py --image_path "$IMG_PATH" --folder_path "$DIR_PATH" --iteration any_trajectory --radius 40 --end_position 30 2 -10 --lr 0.02 --weight_clamp 0.2
```
`--raidus` is the distance from the original camera and the center of the image, i.e., the depth of the center pixel, may need to change to accommodate different images. The original camera position is set to `[radius,0,0]`.

`--end_position` is where the camera trajectory ends at as you like. It need three inputs for the camera position in X, Y, Z. The trajectory will be generated between original camera position to end position and the camera will always face to the center object of the given image.




## Acknowledgement
Thanks for the following wonderful works: [Diffusers](https://huggingface.co/docs/diffusers/index), [Depth Anything](https://github.com/LiheYoung/Depth-Anything), [Depth AnythingV2](https://github.com/DepthAnything/Depth-Anything-V2)..

## Citation

If you find the project is interesting, please cite
   ```sh
@article{you2024nvs,
  title={NVS-Solver: Video Diffusion Model as Zero-Shot Novel View Synthesizer},
  author={You, Meng and Zhu, Zhiyu and Liu, Hui and Hou, Junhui},
  journal={arXiv preprint arXiv:2405.15364},
  year={2024}
}
   ```
