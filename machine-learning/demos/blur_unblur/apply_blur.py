from PIL import Image, ImageFilter
import os

def apply_blur_and_save(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load and process images
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Load the image
            img = Image.open(input_path)

            # Apply blur effect
            blurred_img = img.filter(ImageFilter.GaussianBlur(radius=2))  # You can adjust the radius

            # Save the blurred image
            blurred_img.save(output_path)

    print("Blur applied to images and saved in 'blurred_images' folder.")

if __name__ == "__main__":
    input_folder = "images"        # Folder containing input images
    output_folder = "blurred_images"  # Folder where blurred images will be saved

    apply_blur_and_save(input_folder, output_folder)
