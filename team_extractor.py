#!/usr/bin/env python

import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


PET_SIZE = 48
COORDS_AUTOPLAY = (slice(20, 77), slice(329, 403))
COORDS_HOURGLASS = (slice(5, 35), slice(190, 220))
COORDS_TEAM = (slice(215, 270), slice(330, 640))


def show(*imgs):
    fig, axes = plt.subplots(1, len(imgs), squeeze=False)
    for ax, img in zip(axes[0], imgs):
        if img.shape[-1] == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap='gray')
    plt.show()

def save_mask_icon(frame):
    # From frame 2410
    autoplay = frame[COORDS_AUTOPLAY]
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

    autoplay = autoplay[10:-10, 10:-10]
    mask = mask[10:-10, 10:-10]

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


class TeamExtractor:

    def __init__(self, video_file):
        self.video_file = video_file
        self.video = cv2.VideoCapture(video_file)

        self._load_pets()
        self._load_status()
        self._load_assets()

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

            img = cv2.resize(img, (PET_SIZE, PET_SIZE))[15:, :]
            mask = (img.sum(axis=2) != 0).astype(np.uint8)

            self.pet_imgs[pet_name] = (img, mask)

    def _load_status(self):
        self.status_imgs = {}

    def _load_assets(self):
        self.autoplay = cv2.imread("assets/autoplay_icon.png", cv2.IMREAD_UNCHANGED)
        self.autoplay_mask = self.autoplay[:, :, 3]
        self.autoplay = cv2.cvtColor(self.autoplay, cv2.COLOR_BGR2RGB)

        self.hourglass = cv2.imread("assets/hourglass_icon.png")
        self.hourglass = cv2.cvtColor(self.hourglass, cv2.COLOR_BGR2RGB)

    def get_frame(self, frame_id=None):
        if frame_id is not None:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self.video.read()
        if not ret:
            raise Exception("Couldn't read frame")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def extract_pets(self, frame):
        team = []
        frame = frame[COORDS_TEAM]
        print("Extracting frame ", self.video.get(cv2.CAP_PROP_POS_FRAMES))
        for pet_name, (pet_img, pet_mask) in self.pet_imgs.items():
            res = cv2.matchTemplate(frame, pet_img, cv2.TM_SQDIFF, mask=pet_mask)

            if res[res <= 1.2*res.min()].size >= 10:
                continue

            # threshold = 4e6
            # threshold = 4500 * pet_mask.sum()
            threshold = 1.2*res.min()
            loc = np.where(res < threshold)

            h, w = pet_img.shape[:2]
            rectangles = [(x, y, x+w, y+h) for (y, x) in zip(*loc)]
            rectangles *= 2
            rectangles, weights = cv2.groupRectangles(rectangles, 1)

            # img = frame.copy()
            # print(pet_name, len(rectangles))
            # for (x, y, xw, yh) in rectangles:
            #     cv2.rectangle(img, (x, y), (xw, yh), (0, 255, 0), 2)
            # print([rect[0] for rect in rectangles])
            # plt.imshow(img)
            # plt.show()
            for (x, y, xw, yh) in rectangles:
                team.append((pet_name, x))

        team.sort(key=lambda p: p[1])
        team = list(map(lambda p: p[0], team))
        return team

    def goto_next_battle(self):
        print("Looking for battle")
        while True:
            frame = self.get_frame()
            res = cv2.matchTemplate(frame[COORDS_AUTOPLAY], self.autoplay, cv2.TM_SQDIFF, mask=self.autoplay_mask)
            # if res.min() < 500000:
            if res[res <= 1.2*res.min()].size <= 10:
                print(self.video.get(cv2.CAP_PROP_POS_FRAMES))
                break
        return frame

    def goto_next_turn(self):
        print("Looking for new turn")
        while True:
            frame = self.get_frame()
            res = cv2.matchTemplate(frame[COORDS_HOURGLASS], self.hourglass, cv2.TM_SQDIFF)
            if res.min() < 500000:
                # show(frame)
                break

    def extract_team(self):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 6645)
        frame = self.goto_next_battle()
        while frame is not None:
            pets = self.extract_pets(frame)
            print("List of pets:", pets)
            show(frame)
            self.goto_next_turn()
            frame = self.goto_next_battle()

        self.video.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Path to the video to process")
    args = parser.parse_args()

    team_extractor = TeamExtractor(args.path)
    team_extractor.extract_team()

