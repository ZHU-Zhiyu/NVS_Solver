import os
from moviepy.editor import VideoFileClip

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
input_folder = '/data/zhu_19/NVS_Solver/Assests/dynamic/4dgs/'
output_folder = '/data/zhu_19/NVS_Solver/Assests/dynamic/4dgs_gif/'
gif_size = (400, 240)  # Specify the desired size in pixels (width, height)
convert_folder_to_gif(input_folder, output_folder, gif_size)