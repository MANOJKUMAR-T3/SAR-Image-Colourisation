from PIL import Image
import os

def convert_to_grayscale(image_path, save_path):
    try:
        image = Image.open(image_path).convert('L')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path)
        print(f"Converted {image_path} to grayscale and saved as {save_path}")
    except Exception as e:
        print(f"Failed to convert {image_path}: {e}")


input_dir = r"  "#PATH TO DIRECTORY(COLOURIZED IMAGES)
output_dir = r"  "#PATH TO DIRECTORY(SAR IMAGES)

for root, dirs, files in os.walk(input_dir):
    for filename in files:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(root, filename)
            relative_path = os.path.relpath(root, input_dir)
            output_path = os.path.join(output_dir, relative_path, filename)
            convert_to_grayscale(input_path, output_path)
        else:
            print(f"Skipped {filename} (not a .jpg or .png file)")

