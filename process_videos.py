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
    parser.add_argument('paths', type=str, nargs='+',
                        help="Video ids or paths to a list of video ids to process")
    parser.add_argument('-f', '--nb_finders', type=int, default=2, help="Number of battle finders to run in parallel")
    parser.add_argument('-e', '--nb_extractors', type=int, default=2, help="Number of team extractors to run in parallel")
    parser.add_argument('-d', '--nb_downloaders', type=int, default=2, help="Number of video downloaders to run in parallel")
    parser.add_argument('--download_only', action='store_true', help="Number of video downloaders to run in parallel")
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
            team_files = glob.glob(os.path.join(video_path, 'team_*.png'))
            if not self.download_only and len(team_files) > 10:
                print(f"Video {video_id} seems already processed ! Exiting...")
                self.queue.put(None)
                return

            print(f"Downloading video {video_id}")
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            download_cmd = ["yt-dlp", "--ignore-config", video_url, "-f", "136", "-o", video_file]
            subprocess.run(download_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        else:
            print(f"Video {video_id} already downloaded")
        self.queue.put(video_id)

    def process(self, video_id, nb_finders, nb_extractors):
        video_path = os.path.join("checks", video_id)
        video_file = os.path.join(video_path, "video.mp4")

        # Remove previously extracted teams
        team_files = glob.glob(os.path.join(video_path, 'team_*.png'))
        list(map(os.remove, team_files))

        # Process
        print(f"Processing video {video_id}")
        team_extractor = TeamExtractor(video_file, video_path)
        team_extractor.run(nb_finders=nb_finders, nb_extractors=nb_extractors)

        # Remove video
        os.remove(video_file)

    def process_list(self, paths, nb_finders, nb_extractors, nb_downloaders, download_only):
        video_ids = []
        for path in paths:
            if os.path.isfile(path):
                with open(path, 'r') as file:
                    video_ids.extend(file.read().split('\n')[:-1])
            elif len(path) == 11:    # Video id
                video_ids.append(path)
            else:
                print(f"Arg {path} is not a file nor seems to be a youtube video id")

        self.download_only = download_only
        pool = multiprocessing.Pool(processes=nb_downloaders)
        res = pool.map_async(self.download, video_ids)

        if download_only:
            res.wait()
            return

        downloaded_ids = []
        while len(downloaded_ids) < len(video_ids):
            downloaded_id = self.queue.get()
            downloaded_ids.append(downloaded_id)
            if downloaded_id is not None:
                self.process(downloaded_id, nb_finders, nb_extractors)


if __name__ == '__main__':
    args = parse_args()
    processor = VideoProcessor()
    processor.process_list(args.paths, args.nb_finders, args.nb_extractors, args.nb_downloaders, args.download_only)

