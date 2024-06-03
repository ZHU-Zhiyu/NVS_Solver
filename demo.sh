export CUDA_VISIBLE_DEVICES=5

# ##dynamic image

DIR_PATH=/home/youmeng/youmeng/diffusers_final/example_imgs/dynamic

python svd_interpolate_dyn_img.py --folder_path "$DIR_PATH" --iteration 1 --major_radius 15 --minor_radius 30 --degrees_per_frame 1.0 --lr 0.02 --weight_clamp 0.6
# python svd_interpolate_dyn_img.py --folder_path "$DIR_PATH" --iteration 3 --major_radius 15 --minor_radius 30 --degrees_per_frame 1.0 --inverse True --lr 0.02 --weight_clamp 0.6


# ####multi image
# python svd_interpolate_two_img.py --lr 0.02 --scene scan1 --src1 23 --src2 26 --dataset dtu


###single image


# IMG_PATH=/home/youmeng/youmeng/diffusers_final/example_imgs/single/000001.jpg
# DIR_PATH=/home/youmeng/youmeng/diffusers_final/example_imgs/single/
# python svd_interpolate_single_img.py --image_path "$IMG_PATH" --folder_path "$DIR_PATH" --iteration 3 --inverse True  --major_radius 25 --minor_radius 23 --degrees_per_frame 1.0 --lr 0.02 --weight_clamp 0.2
# python svd_interpolate_single_img.py --image_path "$IMG_PATH" --folder_path "$DIR_PATH" --iteration 1 --major_radius 25 --minor_radius 23 --degrees_per_frame 1.0 --lr 0.02 --weight_clamp 0.2
