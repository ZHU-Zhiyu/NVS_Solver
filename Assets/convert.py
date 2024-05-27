import os
from moviepy.editor import VideoFileClip

import os
import imageio
from PIL import Image
import os
import imageio
def convert_folder_to_gif(input_folder, output_folder, gif_size):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all MP4 files in the input folder
    mp4_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]

    # Convert each MP4 file to GIF
    for mp4_file in mp4_files:
        mp4_path = os.path.join(input_folder, mp4_file)
        gif_file = os.path.splitext(mp4_file)[0] + '.gif'
        gif_path = os.path.join(output_folder, gif_file)
        
        clip = VideoFileClip(mp4_path)
        clip_resized = clip.resize(gif_size)
        clip_resized.write_gif(gif_path)

# Example usage
input_folder = '/data/zhu_19/NVS_Solver/Assests/multi/ivid_multi/'
output_folder = '/data/zhu_19/NVS_Solver/Assests/multi/ivid_multi_01/'
gif_size = (400, 240)  # Specify the desired size in pixels (width, height)
convert_folder_to_gif(input_folder, output_folder, gif_size)


# def speed_up_folder_of_gifs(input_folder, output_folder, speed_factor):
#     # Create the output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # Get a list of all GIF files in the input folder
#     gif_files = [f for f in os.listdir(input_folder) if f.endswith('.gif')]

#     # Process each GIF file
#     for gif_file in gif_files:
#         input_path = os.path.join(input_folder, gif_file)
#         output_path = os.path.join(output_folder, gif_file)

#         # Read the GIF file
#         gif = imageio.mimread(input_path)

#         # Increase the speed of the GIF frames
#         sped_up_frames = [frame for i, frame in enumerate(gif) if i % speed_factor == 0]

#         # Save the sped-up GIF frames as a new GIF
#         imageio.mimsave(output_path, sped_up_frames[:-2])


# Example usage
# input_folder = '/data/zhu_19/NVS_Solver/Assests/multi/ours/'
# output_folder = '/data/zhu_19/NVS_Solver/Assests/multi/ours_01/'
# speed_factor = 2  # Increase speed by a factor of 2
# speed_up_folder_of_gifs(input_folder, output_folder, speed_factor)