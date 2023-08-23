import os

folder_path = "/Users/tushargoyal/Documents/GitHub/mlpr/images"  # Replace this with the actual path to your images folder

# Get a list of all files in the folder
file_list = os.listdir(folder_path)

# Iterate through the files and rename them
for i, filename in enumerate(file_list, start=1):
    old_path = os.path.join(folder_path, filename)
    new_filename = f"{i}.jpg"  # You can adjust the file extension if needed
    new_path = os.path.join(folder_path, new_filename)
    
    # Rename the file
    os.rename(old_path, new_path)
    print(f"Renamed {filename} to {new_filename}")

print("All files have been renamed.")
