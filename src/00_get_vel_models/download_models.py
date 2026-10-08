"""
This script download FWI models from original google drive
For manual download: https://drive.google.com/drive/folders/1IBM_04bPCBnSMO1TMdaps3EHV5dYhYJa?usp=drive_link
"""

import gdown
import os

INPUT_FILE = os.path.join("data","vel_models","drive_file_list.txt")
OUTPUT_FOLDER = os.path.join("data","vel_models","open_fwi")


def read_drive_list(file_path):
    """Lee el archivo y devuelve lista de (name, id)"""
    entries = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if "|" not in line:
            continue
        if "new_name" in line:
            continue
        if "---" in line:
            continue

        name, file_id = [x.strip() for x in line.split("|")]
        entries.append((name, file_id))

    return entries


def download_files(entries, output_folder):
    import os

    os.makedirs(output_folder, exist_ok=True)

    for name, file_id in entries:
        url = f"https://drive.google.com/uc?id={file_id}"
        output_path = os.path.join(output_folder, name)

        print(f"Downloading {name}...")
        gdown.download(url, output_path, quiet=False)


def main():
    entries = read_drive_list(INPUT_FILE)
    print(f"Found {len(entries)} files")

    download_files(entries, OUTPUT_FOLDER)


if __name__ == "__main__":
    main()