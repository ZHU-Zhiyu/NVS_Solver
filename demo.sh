export CUDA_VISIBLE_DEVICES=6

##################POST MODE##################
# ###single image

### path to image
IMG_PATH=example_imgs/single/000001.jpg
### path to data
DIR_PATH=example_imgs/single/
### rotate the camera to right 
python svd_interpolate_single_img.py --image_path "$IMG_PATH" --folder_path "$DIR_PATH" --iteration 1 --major_radius 60 --minor_radius 70 --degrees_per_frame 1.0 --lr 0.02 --weight_clamp 0.2
### rotate the camera to left 
python svd_interpolate_single_img.py --image_path "$IMG_PATH" --folder_path "$DIR_PATH" --iteration 3 --inverse True  --major_radius 60 --minor_radius 70 --degrees_per_frame 1.0 --lr 0.02 --weight_clamp 0.2


### dynamic setting
### path to data
DIR_PATH=example_imgs/dynamic

### rotate the camera to right 
python svd_interpolate_dyn_img.py --folder_path "$DIR_PATH" --iteration 1 --major_radius 15 --minor_radius 30 --degrees_per_frame 1.0 --lr 0.02 --weight_clamp 0.6
### rotate the camera to left 
python svd_interpolate_dyn_img.py --folder_path "$DIR_PATH" --iteration 3 --major_radius 15 --minor_radius 30 --degrees_per_frame 1.0 --inverse True --lr 0.02 --weight_clamp 0.6


#### spare setting
python svd_interpolate_two_img.py --lr 0.02 --scene scan1 --src1 23 --src2 26 --dataset dtu


##################DGS MODE#################

###single image

## path to image
IMG_PATH=example_imgs/single/000001.jpg
### path to data
DIR_PATH=example_imgs/single/
### rotate the camera to right 
python svd_interpolate_single_img_dgs.py --image_path "$IMG_PATH" --folder_path "$DIR_PATH" --iteration dgs_1 --major_radius 60 --minor_radius 70 --degrees_per_frame 1.0 --weight_clamp 0.2
### rotate the camera to left 
python svd_interpolate_single_img_dgs.py --image_path "$IMG_PATH" --folder_path "$DIR_PATH" --iteration dgs_3 --inverse True  --major_radius 60 --minor_radius 70 --degrees_per_frame 1.0 --weight_clamp 0.2


### dynamic setting
# ### path to data
DIR_PATH=example_imgs/dynamic

### rotate the camera to right 
python svd_interpolate_dyn_img_dgs.py --folder_path "$DIR_PATH" --iteration dgs_1 --major_radius 15 --minor_radius 30 --degrees_per_frame 1.0  --weight_clamp 0.6
### rotate the camera to left 
python svd_interpolate_dyn_img_dgs.py --folder_path "$DIR_PATH" --iteration dgs_3 --major_radius 15 --minor_radius 30 --degrees_per_frame 1.0 --inverse True  --weight_clamp 0.6


#### spare setting
python svd_interpolate_two_img_dgs.py  --scene scan1 --src1 23 --src2 26 --dataset dtu
