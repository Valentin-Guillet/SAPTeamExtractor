#!/usr/bin/env python

import argparse
import glob
import multiprocessing
import os
import subprocess
import time

from team_extractor import TeamExtractor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Path to a list of video ids to process")
    parser.add_argument('-n', '--nb_workers', type=int, default=2, help="Number of workers to run in parallel")
    parser.add_argument('-d', '--nb_downloaders', type=int, default=4, help="Number of video downloaders to run in parallel")
    return parser.parse_args()

class VideoProcessor:

    def __init__(self):
        self.queue = multiprocessing.Manager().Queue()

        if not os.path.isdir("checks"):
            os.mkdir("checks")

    def download(self, video_id):
        video_path = os.path.join("checks", video_id)
        video_file = os.path.join(video_path, "video.mp4")
        os.makedirs(video_path, exist_ok=True)

        if not os.path.isfile(video_file):
            print(f"Downloading video {video_id}")
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            download_cmd = ["yt-dlp", "--ignore-config", video_url, "-f", "136", "-o", video_file]
            subprocess.run(download_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        else:
            print(f"Video {video_id} already downloaded")
        self.queue.put(video_id)

    def process(self, video_id, nb_workers):
        video_path = os.path.join("checks", video_id)
        video_file = os.path.join(video_path, "video.mp4")

        # Remove previously extracted teams
        team_files = glob.glob(os.path.join(video_path, 'team_*.png'))
        list(map(os.remove, team_files))

        # Process
        print(f"Processing video {video_id}")
        team_extractor = TeamExtractor(video_file, video_path)
        team_extractor.run(nb_workers=nb_workers)

        # Remove video
        os.remove(video_file)

    def process_list(self, path, nb_workers=2, nb_downloaders=4):
        with open(path, 'r') as file:
            video_ids = file.read().split('\n')[:-1]

        pool = multiprocessing.Pool(processes=nb_downloaders)
        pool.map(self.download, video_ids)

        downloaded_ids = []
        while len(downloaded_ids) < len(video_ids):
            downloaded_id = self.queue.get()
            downloaded_ids.append(downloaded_id)
            self.process(downloaded_id, nb_workers)


if __name__ == '__main__':
    args = parse_args()
    processor = VideoProcessor()
    processor.process_list(args.path, args.nb_workers, args.nb_downloaders)

