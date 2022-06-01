#!/usr/bin/env python

import argparse
import json
import os
import shutil
import subprocess


DATA_FILE = "data.json"
WRONG_COMMITS = ["Bus", "Boar", "Dromedary"]

def download_img(data, path, img_size):
    source = data['image']['source']
    commit = data['image']['commit']
    code = data['image']['unicodeCodePoint'][0]
    code = hex(ord(code))[2:]
    img_type = data['id'].split('-')[0]
    if not img_type.endswith('s'):
        img_type += 's'
    dst_file = f"imgs/{img_type}/{data['name']}.png"

    if data['name'] in WRONG_COMMITS:
        git_dir = os.path.join(path, source)
        git_cmd = ["git", "-C", git_dir, "switch", "--detach", commit]
        subprocess.run(git_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if source == "noto-emoji":
        file_name = f"emoji_u{code}.svg"
        src_file = os.path.join(path, source, "svg", file_name)

    elif source == "fxemoji":
        file_name = f"u{code.upper()}-{data['image']['name']}.svg"
        src_file = os.path.join(path, source, "svgs/FirefoxEmoji", file_name)

    elif source == "twemoji":
        file_name = f"{code}.svg"
        src_file = os.path.join(path, source, "assets/svg", file_name)

    print(f"Processing {data['name']} ({src_file} to {dst_file})")
    # convert_cmd = ["convert", "-background", "none", "-size", f"{img_size}x{img_size}", src_file, dst_file]
    branch_name = subprocess.run
    convert_cmd = ["inkscape", "-w", str(img_size), "-h", str(img_size), src_file, "-o", dst_file]
    subprocess.run(convert_cmd, stderr=subprocess.DEVNULL)

    if data['name'] in WRONG_COMMITS:
        git_cmd = ["git", "-C", git_dir, "branch", "--format=%(refname:short)"]
        branches = subprocess.run(git_cmd, capture_output=True).stdout.decode().split('\n')[:-1]
        branch_name = next(filter(lambda branch: 'detached' not in branch, branches))
        git_cmd = ["git", "-C", git_dir, "switch", branch_name]
        subprocess.run(git_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def download_imgs(file_name, path, img_size):
    with open(file_name, 'r') as file:
        data = json.load(file)

    for pet_data in data['pets'].values():
        os.makedirs("imgs/pets", exist_ok=True)
        download_img(pet_data, path, img_size)

    for food_data in data['foods'].values():
        os.makedirs("imgs/foods", exist_ok=True)
        download_img(food_data, path, img_size)

    for status_data in data['statuses'].values():
        os.makedirs("imgs/status", exist_ok=True)
        download_img(status_data, path, img_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Path to directory containing the github repository of noto-emoji, fxemoji and twemoji")
    parser.add_argument('-s', '--size', type=int, default=120, help="Size of the final png")
    args = parser.parse_args()
    download_imgs(DATA_FILE, args.path, args.size)

