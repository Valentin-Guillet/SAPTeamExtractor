#!/usr/bin/env python

import argparse
import logging
import multiprocessing
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image


PET_SIZE = 100


def extend(coords, dx, dy=None):
    if dy is None:
        dy = dx
    x, y = coords
    return (slice(x.start-dx, x.stop+dx), slice(y.start-dy, y.stop+dy))

COORDS_AUTOPLAY = (slice(58, 137), slice(674, 789))
COORDS_AUTOPLAY_AREA = extend(COORDS_AUTOPLAY, 15)

COORDS_HOURGLASS = (slice(25, 56), slice(401, 423))
COORDS_HOURGLASS_AREA = extend(COORDS_HOURGLASS, 15)

COORDS_TEAM = (slice(430, 600), slice(660, 1275))
COORDS_ATTACK = [(slice(546, 595), slice(670+120*i, 719+120*i)) for i in range(5)]
COORDS_LIFE = [(slice(546, 595), slice(729+120*i, 778+120*i)) for i in range(5)]

COORDS_PETS = [(slice(COORDS_LIFE[spot][0].start - 130, COORDS_LIFE[spot][0].start - 3),
                slice(COORDS_ATTACK[spot][1].start - 8, COORDS_LIFE[spot][1].stop + 8))
               for spot in range(5)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Path to the video to process")
    parser.add_argument('-n', '--nb_workers', type=int, help="Number of workers to run in parallel")
    return parser.parse_args()

def show(*imgs):
    fig, axes = plt.subplots(1, len(imgs), squeeze=False)
    for ax, img in zip(axes[0], imgs):
        if img.shape[-1] == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap='gray')
    plt.show()

def save_autoplay_icon(frame):
    # From frame 2800
    autoplay = frame[COORDS_AUTOPLAY_AREA]
    contour = cv2.Canny(autoplay, 100, 200)
    h, w = contour.shape
    tmp_mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(contour, tmp_mask, (1, 1), None, flags=cv2.FLOODFILL_MASK_ONLY)
    tmp_mask = tmp_mask[1:-1, 1:-1]

    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(tmp_mask, mask, (h//2, w//2), None, flags=cv2.FLOODFILL_MASK_ONLY)

    mask2 = mask.copy()
    for i in range(2, h):
        for j in range(2, w):
            if not mask[i-1, j] or not mask[i+1, j] or not mask[i, j-1] or not mask[i, j+1]:
                mask2[i, j] = False
    mask = mask2.copy()
    mask = np.expand_dims(mask[1:-1, 1:-1], axis=2)

    autoplay = autoplay[15:-15, 15:-15]
    mask = mask[15:-15, 15:-15]

    autoplay *= mask
    autoplay = np.c_[autoplay, 255*np.ones((*autoplay.shape[:2], 1, ))]
    autoplay[:, :, 3] *= mask[:, :, 0]
    autoplay = autoplay.astype(np.uint8)
    os.makedirs("assets", exist_ok=True)
    img = Image.fromarray(autoplay)
    img.save("assets/autoplay_icon.png")

def add_border(img, mask):
    mask_copy = mask.copy()
    for i in range(PET_SIZE):
        for j in range(PET_SIZE):
            if mask[i, j]:
                continue
            if (0 < i and mask[i-1, j] or
                    i < PET_SIZE-1 and mask[i+1, j] or
                    0 < j and mask[i, j-1] or
                    j < PET_SIZE-1 and mask[i, j+1]):
                img[i, j] = (255, 255, 255)
                mask_copy[i, j] = True
    return img, mask_copy

def get_found_score(frame, img, mask):
    res = cv2.matchTemplate(frame, img, cv2.TM_SQDIFF, mask=mask)
    nb_peaks = (res < 1.2*res.min()).sum()
    _, _, loc, _ = cv2.minMaxLoc(res)
    found_img = frame[loc[1]:loc[1]+img.shape[0], loc[0]:loc[0]+img.shape[1]]

    close_pixels = (np.abs(found_img.astype(np.int16) - img).mean(axis=2) < 10) * mask
    score = 100 * close_pixels.sum() / mask.size
    return score, nb_peaks


class TeamExtractor:

    def __init__(self, video_file):
        self.video_file = video_file
        self.video_length = int(cv2.VideoCapture(video_file).get(cv2.CAP_PROP_FRAME_COUNT))

        self._load_pets()
        self._load_status()
        self._load_assets()

        self.queue = multiprocessing.Queue()

        self.logger = logging.getLogger('main')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())

    def _load_pets(self):
        self.pet_imgs = {}
        for file in os.listdir("imgs/pets"):
            pet_name = file[:-4]
            img = cv2.imread(f"imgs/pets/{file}", cv2.IMREAD_UNCHANGED)
            if img.dtype == 'uint16':
                img = (img // 256).astype(np.uint8)
            mask = img[:, :, 3] > 0
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img *= mask[:, :, np.newaxis]

            img = cv2.resize(img, (PET_SIZE, PET_SIZE))[30:, :]
            mask = (img.sum(axis=2) != 0).astype(np.uint8)

            self.pet_imgs[pet_name] = (img, mask)

    def _load_status(self):
        self.status_imgs = {}
        for file in os.listdir("imgs/status"):
            status_name = file[:-4]
            img = cv2.imread(f"imgs/status/{file}", cv2.IMREAD_UNCHANGED)
            if img.dtype == 'uint16':
                img = (img // 256).astype(np.uint8)
            mask = img[:, :, 3] > 0
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img *= mask[:, :, np.newaxis]
            img = cv2.resize(img, (PET_SIZE // 2, PET_SIZE // 2))
            mask = (img.sum(axis=2) != 0).astype(np.uint8)

            self.status_imgs[status_name] = (img, mask)

    def _load_assets(self):
        self.autoplay = cv2.imread("assets/autoplay_icon.png", cv2.IMREAD_UNCHANGED)
        self.autoplay_mask = self.autoplay[:, :, 3]
        self.autoplay = cv2.cvtColor(self.autoplay, cv2.COLOR_BGR2RGB)

        self.hourglass = cv2.imread("assets/hourglass_icon.png")
        self.hourglass = cv2.cvtColor(self.hourglass, cv2.COLOR_BGR2RGB)

    def get_frame(self, capture, frame_id=None):
        if frame_id is not None:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = capture.read()
        if not ret:
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def find_spots(self, frame):
        spots = []
        for spot in range(5):
            attack = frame[COORDS_ATTACK[spot]]
            m = np.repeat(attack.mean(axis=2)[..., np.newaxis], 3, axis=2)
            grey_pixels = (np.abs(m - attack).sum(axis=2) <= 20).sum()

            if grey_pixels >= 500:
                spots.append(spot)
        return spots

    def extract_status(self, frame, spots):
        all_status = []
        for spot in range(5):
            if spot not in spots:
                all_status.append(None)
                continue

            pet_area = frame[COORDS_PETS[spot]]
            for status_name, (status_img, status_mask) in self.status_imgs.items():
                for size in range(30, 50, 5):
                    resized_status_img = cv2.resize(status_img, (size, size))
                    resized_status_mask = (resized_status_img.sum(axis=2) != 0).astype(np.uint8)

                    score, nb_peaks = get_found_score(pet_area, resized_status_img, resized_status_mask)
                    if score > 15 and nb_peaks < 30:
                        all_status.append(status_name)
                        break

                else:
                    continue
                break

            else:
                all_status.append("Nothing")

        return all_status

    def extract_pets(self, frame, spots, status):
        team = []
        for spot in range(5):
            if spot not in spots:
                team.append(None)
                continue

            pet_area = frame[COORDS_PETS[spot]]
            scores = {}
            for pet_name, (pet_img, pet_mask) in self.pet_imgs.items():
                scores[pet_name], _ = get_found_score(pet_area, pet_img, pet_mask)

            team.append(max(scores, key=scores.get))

        return team

    def extract_team(self, frame):
        spots = self.find_spots(frame)
        status = self.extract_status(frame, spots)
        pets = self.extract_pets(frame, spots, status)
        return pets, status

    def goto_next_battle(self, capture):
        while True:
            frame = self.get_frame(capture)
            if frame is None:
                return None, -1

            res = cv2.matchTemplate(frame[COORDS_AUTOPLAY_AREA], self.autoplay, cv2.TM_SQDIFF, mask=self.autoplay_mask)
            if (res <= 1.2*res.min()).sum() <= 20:
                break

        frame_nb = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        return frame, frame_nb

    def goto_next_turn(self, capture):
        while True:
            frame = self.get_frame(capture)
            if frame is None:
                return

            res = cv2.matchTemplate(frame[COORDS_HOURGLASS_AREA], self.hourglass, cv2.TM_SQDIFF)
            if (res <= 1.2*res.min()).sum() <= 20:
                self.get_frame(capture)   # Wait one frame to pass black screen
                break

    def find_battles(self, worker_id, init_frame, end_frame):
        self.logger.info(f"[WORKER {worker_id}] Running between frames {init_frame} and {end_frame}")
        capture = cv2.VideoCapture(self.video_file)
        capture.set(cv2.CAP_PROP_POS_FRAMES, init_frame)

        # If in battle, skip it
        frame = self.get_frame(capture)
        res = cv2.matchTemplate(frame[COORDS_AUTOPLAY_AREA], self.autoplay, cv2.TM_SQDIFF, mask=self.autoplay_mask)
        if (res <= 1.2*res.min()).sum() <= 20:
            self.logger.info(f"[WORKER {worker_id}] Starting in the middle of a battle ! Skipping...")
            self.goto_next_turn(capture)

        frame, frame_nb = self.goto_next_battle(capture)
        while frame is not None and frame_nb < end_frame:
            self.logger.info(f"[WORKER {worker_id}] Battle found ! Putting in queue")
            self.queue.put(frame)
            self.goto_next_turn(capture)
            frame, frame_nb = self.goto_next_battle(capture)

        self.logger.info(f"[WORKER {worker_id}] Done !")
        self.queue.put(worker_id)
        capture.release()

    def extract_teams(self, nb_workers):
        self.logger.info("[EXTRACTOR] Initializing")
        workers_done = []
        frame = self.queue.get()
        while type(frame) is int:
            workers_done.append(frame)
            frame = self.queue.get()

        while len(workers_done) < nb_workers:
            self.logger.info("[EXTRACTOR] Processing frame")
            pets, status = self.extract_team(frame)

            self.logger.info("[EXTRACTOR] List of pets:" + str(pets))
            self.logger.info("[EXTRACTOR] List of status:" + str(status))
            frame = self.queue.get()
            while type(frame) is int:
                self.logger.info(f"[EXTRACTOR] Worker {frame} done")
                workers_done.append(frame)
                if len(workers_done) < nb_workers:
                    frame = self.queue.get()
                else:
                    break

        self.logger.info("[EXTRACTOR] Extractor done !")

    def run_sync(self):
        capture = cv2.VideoCapture(self.video_file)
        capture.set(cv2.CAP_PROP_POS_FRAMES, 2530)
        frame, frame_nb = self.goto_next_battle(capture)
        while frame is not None:
            self.logger.info(f"Frame {frame_nb}: battle found")
            pets, status = self.extract_team(frame)

            self.logger.info("List of pets:" + str(pets))
            self.logger.info("List of status:" + str(status))
            self.goto_next_turn(capture)
            show(frame)
            frame, frame_nb = self.goto_next_battle(capture)

        capture.release()

    def run(self, nb_workers=5):
        if nb_workers == 1:
            self.run_sync()
            return

        frame_limits = [(i*self.video_length) // nb_workers for i in range(nb_workers+1)]

        battle_finders = []
        for i in range(nb_workers):
            proc = multiprocessing.Process(target=self.find_battles, args=(i, *frame_limits[i:i+2]))
            battle_finders.append(proc)

        team_extractor = multiprocessing.Process(target=self.extract_teams, args=(nb_workers, ))

        for proc in battle_finders:
            proc.start()
        team_extractor.start()

        for proc in battle_finders:
            proc.join()
        team_extractor.join()


if __name__ == '__main__':
    args = parse_args()
    team_extractor = TeamExtractor(args.path)
    team_extractor.run(nb_workers=args.nb_workers)

