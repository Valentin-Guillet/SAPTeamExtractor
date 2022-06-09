#!/usr/bin/env python

import argparse
import json
import os
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image


DATA_FILE = "data.json"
WRONG_COMMITS = ["Bus", "Boar", "Dromedary"]

SAP_WIKI_URL = {"Tabby Cat": "4/4c/TabbyCat.png",
                "Garlic Armor": "c/cc/Garlic.png", }


def download_from_wiki(img_name, dst_file, img_size):
    print(f"Processing {img_name} (from wiki to {dst_file})")
    revision = SAP_WIKI_URL[img_name]
    img_url = f"https://static.wikia.nocookie.net/superautopets/images/{revision}"
    download_cmd = ["wget", img_url, '-O', dst_file]
    subprocess.run(download_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    img = cv2.imread(dst_file, cv2.IMREAD_UNCHANGED)
    h, w, _ = img.shape
    if h > w:
        before_col = np.zeros((h, (h - w) // 2, 4), np.uint8)
        after_col = np.zeros((h, (h - w) // 2, 4), np.uint8)
        img = np.hstack((before_col, img, after_col))

    elif h < w:
        before_row = np.zeros(((w - h), w // 2, 4), np.uint8)
        after_row = np.zeros(((w - h), w // 2, 4), np.uint8)
        img = np.vstack((before_col, img, after_col))

    # BGR to RGB
    img = img[..., (2, 1, 0, 3)]
    img = cv2.resize(img, (img_size, img_size))
    img = Image.fromarray(img)
    img.save(dst_file)

def download_img(data, path, img_size):
    source = data['image']['source']
    commit = data['image']['commit']
    code = data['image']['unicodeCodePoint']
    code = '_'.join([hex(ord(c))[2:] for c in code])
    img_type = data['id'].split('-')[0]
    if not img_type.endswith('s'):
        img_type += 's'
    dst_file = f"imgs/{img_type}/{data['name']}.png"

    if data['name'] == "Garlic Armor":
        download_from_wiki(data['name'], f"imgs/{img_type}/{data['name']}_alt.png", img_size)

    elif data['name'] in SAP_WIKI_URL:
        download_from_wiki(data['name'], dst_file, img_size)
        return

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
    result = subprocess.run(convert_cmd, capture_output=True)
    if not os.path.isfile(dst_file):
        print("An error occured !\n", result.stderr.decode())
        exit()

    if data['name'] in WRONG_COMMITS:
        git_cmd = ["git", "-C", git_dir, "branch", "--format=%(refname:short)"]
        branches = subprocess.run(git_cmd, capture_output=True).stdout.decode().split('\n')[:-1]
        branch_name = next(filter(lambda branch: 'detached' not in branch, branches))
        git_cmd = ["git", "-C", git_dir, "switch", branch_name]
        subprocess.run(git_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def download_imgs(file_name, path, img_size):
    with open(file_name, 'r') as file:
        data = json.load(file)

    os.makedirs("imgs/pets", exist_ok=True)
    for pet_data in data['pets'].values():
        download_img(pet_data, path, img_size)

    os.makedirs("imgs/status", exist_ok=True)
    for status_data in data['statuses'].values():
        download_img(status_data, path, img_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Path to directory containing the github repository of noto-emoji, fxemoji and twemoji")
    parser.add_argument('-s', '--size', type=int, default=120, help="Size of the final png")
    args = parser.parse_args()
    download_imgs(DATA_FILE, args.path, args.size)

