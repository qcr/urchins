import os
import argparse
import cv2


def extract_frames(video_path, output_folder, fps=None):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not video.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    # Create a folder for the current video frames
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_folder = os.path.join(output_folder, video_name)
    os.makedirs(output_video_folder, exist_ok=True)

    # Initialize frame count
    frame_count = 0

    # Read until video is completed
    while True:
        # Capture frame-by-frame
        ret, frame = video.read()

        # If frame is read correctly, save it
        if ret:
            frame_count += 1
            # Save frame as image with name including video file name and frame number
            frame_name = f"{video_name}_frame_{frame_count:05d}.jpg"  # Format frame number with leading zeros
            frame_path = os.path.join(output_video_folder, frame_name)
            cv2.imwrite(frame_path, frame)
        else:
            break

        # Skip frames if needed
        if fps is not None:
            for _ in range(int(video.get(cv2.CAP_PROP_FPS) / fps) - 1):
                video.read()

    # Release the video capture object
    video.release()

    print(f"Frames extracted from {video_path}: {frame_count}")
    print(f"Frames saved in: {output_video_folder}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('source_folder', type=str, help='Path to source folder containing video files')
    parser.add_argument('dest_folder', type=str, help='Path to destination folder for extracted frames')
    parser.add_argument('--fps', type=float, default=None, help='Frames per second (FPS) to extract frames from (default: video FPS)')
    args = parser.parse_args()

    # Iterate over all files in the source folder
    for filename in os.listdir(args.source_folder):
        if filename.endswith(('.mp4', '.avi', '.mov', '.MP4')):  # Check if file is a video file
            video_path = os.path.join(args.source_folder, filename)
            extract_frames(video_path, args.dest_folder, args.fps)

if __name__ == '__main__':
    main()
