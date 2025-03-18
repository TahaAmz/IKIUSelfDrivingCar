import os

def modify_files(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith('.txt'):
            with open(filepath, 'r') as file:
                lines = file.readlines()
            with open(filepath, 'w') as file:
                for line in lines:
                    if line.startswith('3'):
                        line = '11 ' + line[2:]
                    file.write(line)
            print(f"Modified content of {filename}")

directory_path = r"C:\Users\asus\Desktop\raw_imgs\S40_40_wl_pos4"
modify_files(directory_path)
