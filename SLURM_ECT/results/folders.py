import os

num_folders = 116

current_directory = os.getcwd()

for i in range(1, num_folders + 1):
    folder_name = os.path.join(current_directory, str(i))
    os.makedirs(folder_name, exist_ok=True)

print(f"{num_folders} has been created in {current_directory}.")
