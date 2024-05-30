import cv2
import sys
import os

def resize_image(input_image_path, export_path):
    # Load the image
    image = cv2.imread(input_image_path)
    
    # Get the original image dimensions
    height, width = image.shape[:2]
    
    # Calculate the aspect ratio
    aspect_ratio = width / height
    
    # Calculate the new height based on the aspect ratio
    new_height = int(640 / aspect_ratio)
    
    # Resize the image
    resized_image = cv2.resize(image, (640, new_height))
    
    # Extract the filename and extension from the input image path
    filename, extension = os.path.splitext(os.path.basename(input_image_path))
    
    # Construct the output image path with the new filename and export directory
    output_image_path = os.path.join(export_path, f"{filename}_resized{extension}")
    
    # Save the resized image
    cv2.imwrite(output_image_path, resized_image)
    
    print(f"Resized image saved to {output_image_path}")

if __name__ == "__main__":
    # Parse arguments
    input_image_path = sys.argv[1]
    export_directory = sys.argv[2]
    
    # Check if the export directory exists, if not create it
    os.makedirs(export_directory, exist_ok=True)
    
    # Resize the image
    resize_image(input_image_path, export_directory)
