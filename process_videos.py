#!/usr/bin/env python

import argparse
import multiprocessing
import os
import subprocess
import time

from team_extractor import TeamExtractor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Path to a list of video ids to process")
    parser.add_argument('-n', '--nb_workers', type=int, help="Number of workers to run in parallel")
    return parser.parse_args()

def process(video_id):
    print(f"Processing video {video_id}")
    video_path = os.path.join("checks", video_id)
    video_file = os.path.join(video_path, "video.mp4")
    os.makedirs(video_path, exist_ok=True)

    # Download
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    download_cmd = ["yt-dlp", "--ignore-config", video_url, "-f", "136", "-o", video_file]
    subprocess.run(download_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Process
    team_extractor = TeamExtractor(video_file, video_path)
    team_extractor.run(nb_workers=1)

    # Remove video
    os.remove(video_file)

def process_list(path, nb_workers=2):
    if not os.path.isdir("checks"):
        os.mkdir("checks")

    processed_ids = os.listdir("./checks/")
    with open(path, 'r') as file:
        video_ids = file.read().split('\n')[:-1]

    ids_to_process = [id for id in video_ids if id not in processed_ids]
    pool = multiprocessing.Pool(processes=nb_workers)
    pool.map(process, ids_to_process)


if __name__ == '__main__':
    args = parse_args()
    process_list(args.path, args.nb_workers)

