#!/usr/bin/env python

import argparse
import logging
import multiprocessing
import os

import cv2
import matplotlib.pyplot as plt
import matplotlib.widgets as wdg
import numpy as np

from PIL import Image


PET_SIZE = 100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Path to the video to process")
    parser.add_argument('-o', '--output', default=None, type=str, help="Dir in which to save output")
    parser.add_argument('-n', '--nb_workers', type=int, help="Number of workers to run in parallel")
    return parser.parse_args()

def extend(coords, dx, dy=None):
    if dy is None:
        dy = dx
    x, y = coords
    return (slice(x.start-dx, x.stop+dx), slice(y.start-dy, y.stop+dy))

def show(*imgs):
    fig, axes = plt.subplots(1, len(imgs), squeeze=False)
    for ax, img in zip(axes[0], imgs):
        ax.axis('off')
        if img.shape[-1] == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap='gray')
    plt.show()

def save_autoplay_icon(frame):
    # From frame 2800
    autoplay = frame[TeamExtractor.COORDS["autoplay_area"]]
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

def save_lvl_icon(frame):
    # Save LVL writing from spot 2 on frame
    coords_lvl = (slice(414, 436), slice(976, 1004))
    lvl_icon = frame[coords_lvl]

    os.makedirs("assets", exist_ok=True)
    img = Image.fromarray(lvl_icon)
    img.save("assets/lvl_icon.png")

def save_xp_icon(frame):
    # Save XP bar icon from frame where team is low (before SAP update) on spot 2
    coords_xp_2 = (slice(469, 485), slice(974, 1024))
    xp_bar = frame[coords_xp_2]

def search_canny(img, init_min=400, init_max=800):
    fig = plt.figure("canny_search")
    ax = fig.subplots()
    ax.imshow(cv2.Canny(img, init_min, init_max), cmap='gray')

    axmin = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    axmax = fig.add_axes([0.25, 0.01, 0.65, 0.03])
    slider_min = wdg.Slider(ax=axmin, label='Min value', valmin=50, valmax=1500, valinit=init_min, valstep=50)
    slider_max = wdg.Slider(ax=axmax, label='Max value', valmin=50, valmax=1500, valinit=init_max, valstep=50)

    def update(val):
        ax.cla()
        ax.imshow(cv2.Canny(img, slider_min.val, slider_max.val), cmap='gray')
        fig.canvas.draw()

    slider_min.on_changed(update)
    slider_max.on_changed(update)
    plt.show()


class ImgStruct:
    def __init__(self, img):
        mask = (img.sum(axis=2) != 0).astype(np.uint8)
        self.shape = img.shape[:2]
        self.init_shape = self.shape

        self.resized_imgs = {self.shape: img}
        self.resized_masks = {self.shape: mask}
        self.contours = {}

    @property
    def height(self):
        return self.shape[0]

    @property
    def width(self):
        return self.shape[1]

    @property
    def img(self):
        return self.resized_imgs[self.shape]

    @property
    def mask(self):
        return self.resized_masks[self.shape]

    def resize(self, new_shape):
        if new_shape not in self.resized_imgs:
            resized_img = cv2.resize(self.resized_imgs[self.init_shape], new_shape)
            self.resized_imgs[new_shape] = resized_img
            self.resized_masks[new_shape] = (resized_img.sum(axis=2) != 0).astype(np.uint8)

        self.shape = new_shape

    def _compute_contours(self):
        h, w = self.shape[:2]
        contours = np.zeros((h+2, w+2), dtype=np.bool_)
        contours[1:h+1, 1:w+1] = cv2.Canny(self.img, 400, 800)

        all_contours = [contours[1:h+1, 1:w+1], contours[:h, 1:w+1], contours[2:, 1:w+1],
                        contours[1:h+1, :w], contours[1:h+1, 2:]]
        all_nb_pixels = [(contour != 0).sum() for contour in all_contours]

        return all_contours, all_nb_pixels

    def get_contours(self):
        if self.shape not in self.contours:
            self.contours[self.shape] = self._compute_contours()
        return self.contours[self.shape]


class TeamExtractor:

    COORDS = {"autoplay": (slice(58, 137), slice(674, 789)),
              "hourglass": (slice(25, 56), slice(401, 423)),
            }

    COORDS["autoplay_area"] = extend(COORDS["autoplay"], 15)
    COORDS["hourglass_area"] = extend(COORDS["hourglass"], 15)

    COORDS["attack"] = []
    COORDS["life"] = []
    COORDS["pets"] = []

    @classmethod
    def _update_coords(cls, height):
        cls.COORDS["team"] = (slice(height, height+210), slice(660, 1275))

        h = height
        for i in range(5):
            cls.COORDS["attack"].append((slice(h+152, h+201), slice(670+120*i, 719+120*i)))
            cls.COORDS["life"].append((slice(h+152, h+201), slice(729+120*i, 778+120*i)))
            cls.COORDS["pets"].append((slice(h+22, h+149), slice(662+120*i, 786+120*i)))

    def __init__(self, video_file, output_path):
        self.video_file = video_file
        self.video_length = int(cv2.VideoCapture(video_file).get(cv2.CAP_PROP_FRAME_COUNT))
        self.output_path = output_path

        self._load_pets()
        self._load_status()
        self._load_assets()

        self.queue = multiprocessing.Queue()

        self.logger = logging.getLogger('main')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())

        self.team_extracted = 0

    def _load_pets(self):
        self.pets = {}
        for file in os.listdir("imgs/pets"):
            pet_name = file[:-4]
            img = cv2.imread(f"imgs/pets/{file}", cv2.IMREAD_UNCHANGED)
            if img.dtype == 'uint16':
                img = (img // 256).astype(np.uint8)
            mask = img[:, :, 3] > 0
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img *= mask[:, :, np.newaxis]

            # Remove the head position from the mask to account for hats
            if pet_name == "Mosquito":
                # Mosquito has the hat very low, remove the precise area
                img[30:75, 20:80] = 0
                img = cv2.resize(img, (PET_SIZE, PET_SIZE))
            else:
                # Other pets just have the hat on top of them: mask the first rows
                img = cv2.resize(img, (PET_SIZE, PET_SIZE))[30:, :]

            self.pets[pet_name] = ImgStruct(img)

    def _load_status(self):
        self.status = {}
        for file in os.listdir("imgs/status"):
            status_name = file[:-4]
            img = cv2.imread(f"imgs/status/{file}", cv2.IMREAD_UNCHANGED)
            if img.dtype == 'uint16':
                img = (img // 256).astype(np.uint8)
            mask = img[:, :, 3] > 0
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img *= mask[:, :, np.newaxis]
            img = cv2.resize(img, (PET_SIZE // 2, PET_SIZE // 2))
            self.status[status_name] = ImgStruct(img)

    def _load_assets(self):
        self.autoplay = cv2.imread("assets/autoplay_icon.png", cv2.IMREAD_UNCHANGED)
        self.autoplay_mask = self.autoplay[:, :, 3]
        self.autoplay = cv2.cvtColor(self.autoplay, cv2.COLOR_BGR2RGB)

        self.hourglass = cv2.imread("assets/hourglass_icon.png")
        self.hourglass = cv2.cvtColor(self.hourglass, cv2.COLOR_BGR2RGB)

        self.lvl = cv2.imread("assets/lvl_icon.png")
        self.lvl = cv2.cvtColor(self.lvl, cv2.COLOR_BGR2RGB)

        self.xp_bars = []
        for i in range(len(os.listdir("assets/XP/"))):
            xp_bar = cv2.imread("assets/XP/xp_icon_0.png")
            xp_bar = cv2.cvtColor(xp_bar, cv2.COLOR_BGR2RGB)
            self.xp_bars.append(xp_bar)

    def _load_whole_pets(self):
        self.whole_pet_imgs = {}
        for file in os.listdir("imgs/pets"):
            pet_name = file[:-4]
            img = cv2.imread(f"imgs/pets/{file}", cv2.IMREAD_UNCHANGED)
            if img.dtype == 'uint16':
                img = (img // 256).astype(np.uint8)
            mask = img[:, :, 3] > 0
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img *= mask[:, :, np.newaxis]
            img = cv2.resize(img, (PET_SIZE, PET_SIZE))
            self.whole_pet_imgs[pet_name] = ImgStruct(img)

    def get_frame(self, capture, frame_id=None):
        if frame_id is not None:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = capture.read()
        if not ret:
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def get_team_height(self, frame):
        res = cv2.matchTemplate(frame, self.lvl, cv2.TM_SQDIFF)
        _, _, loc, _ = cv2.minMaxLoc(res)
        return loc[1] - 20

    def find_spots(self, frame):
        spots = []
        for spot in range(5):
            attack = frame[self.COORDS["attack"][spot]]
            m = np.repeat(attack.mean(axis=2)[..., np.newaxis], 3, axis=2)
            grey_pixels = (np.abs(m - attack).sum(axis=2) <= 20).sum()

            if grey_pixels >= 500:
                spots.append(spot)
        return spots

    def get_pet_score(self, frame, pet):
        res = cv2.matchTemplate(frame, pet.img, cv2.TM_SQDIFF, mask=pet.mask)
        _, _, loc, _ = cv2.minMaxLoc(res)
        found_img = frame[loc[1]:loc[1]+pet.shape[0], loc[0]:loc[0]+pet.shape[1]]

        close_pixels = (np.abs(found_img.astype(np.int16) - pet.img).mean(axis=2) < 12) * pet.mask
        score = 100 * close_pixels.sum() / pet.mask.size

        return score

    def extract_pets(self, frame, spots):
        team = []
        for spot in range(5):
            if spot not in spots:
                team.append(None)
                continue

            pet_area = frame[self.COORDS["pets"][spot]]
            scores = {}
            for pet_name, pet in self.pets.items():
                scores[pet_name] = self.get_pet_score(pet_area, pet)

            team.append(max(scores, key=scores.get))

        return team

    def get_status_score(self, frame, status, shape):
        status.resize(shape)
        res = cv2.matchTemplate(frame, status.img, cv2.TM_SQDIFF, mask=status.mask)
        _, _, loc, _ = cv2.minMaxLoc(res)
        found_img = frame[loc[1]:loc[1]+shape[0], loc[0]:loc[0]+shape[1]]

        # Closeness score: nb of pixels whose RGB values are close to status img (to maximize)
        close_pixels = (np.abs(found_img.astype(np.int16) - status.img).mean(axis=2) < 12) * status.mask
        closeness_score = 100 * close_pixels.sum() / status.mask.size

        # Peak score: nb of location in the frame that results in almost the minimum of score
        # found in the convolution (to minimize)
        nb_peaks = (res < 1.2*res.min()).sum()

        # Nb of contours in common with the status img (to maximize)
        found_contours = cv2.Canny(found_img, 400, 800).view(np.bool_)
        contours_score = [100 * (contours * found_contours).sum() / size
                          for (contours, size) in zip(*status.get_contours())]
        best_contours_score = max(contours_score)

        return closeness_score, nb_peaks, best_contours_score

    def extract_status(self, frame, pets, spots):
        all_status = []
        for spot in range(5):
            if spot not in spots:
                all_status.append(None)
                continue

            pet_area = frame[self.COORDS["pets"][spot]]
            for status_name, status in self.status.items():
                for size in range(25, 50, 5):
                    closeness_score, nb_peaks, contours_score = self.get_status_score(pet_area, status, (size, size))

                    if closeness_score > 15 and nb_peaks < 20 and contours_score > 35:
                        all_status.append(status_name)
                        break

                else:
                    continue
                break

            else:
                all_status.append("Nothing")

        return all_status

    def extract_team(self, frame):
        if self.team_extracted == 0:
            team_height = self.get_team_height(frame)
            self._update_coords(team_height)

        spots = self.find_spots(frame)
        pets = self.extract_pets(frame, spots)
        status = self.extract_status(frame, pets, spots)
        self.team_extracted += 1
        return pets, status

    def goto_next_battle(self, capture):
        while True:
            frame = self.get_frame(capture)
            if frame is None:
                return None, -1

            res = cv2.matchTemplate(frame[self.COORDS["autoplay_area"]], self.autoplay, cv2.TM_SQDIFF, mask=self.autoplay_mask)
            if (res <= 1.2*res.min()).sum() <= 20:
                break

        frame_nb = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        return frame, frame_nb

    def goto_next_turn(self, capture):
        while True:
            frame = self.get_frame(capture)
            if frame is None:
                return

            res = cv2.matchTemplate(frame[self.COORDS["hourglass_area"]], self.hourglass, cv2.TM_SQDIFF)
            if (res <= 1.2*res.min()).sum() <= 20:
                self.get_frame(capture)   # Wait one frame to pass black screen
                break

    def find_battles(self, worker_id, init_frame, end_frame):
        self.logger.info(f"[WORKER {worker_id}] Running between frames {init_frame} and {end_frame}")
        capture = cv2.VideoCapture(self.video_file)
        capture.set(cv2.CAP_PROP_POS_FRAMES, init_frame)

        # If in battle, skip it
        frame = self.get_frame(capture)
        res = cv2.matchTemplate(frame[self.COORDS["autoplay_area"]], self.autoplay, cv2.TM_SQDIFF, mask=self.autoplay_mask)
        if (res <= 1.2*res.min()).sum() <= 20:
            self.logger.info(f"[WORKER {worker_id}] Starting in the middle of a battle ! Skipping...")
            self.goto_next_turn(capture)

        frame, frame_nb = self.goto_next_battle(capture)
        while frame is not None and frame_nb < end_frame:
            self.logger.info(f"[WORKER {worker_id}] Battle found ! Putting in queue")
            self.queue.put((frame, frame_nb))
            self.goto_next_turn(capture)
            frame, frame_nb = self.goto_next_battle(capture)

        self.logger.info(f"[WORKER {worker_id}] Done !")
        self.queue.put((worker_id, -1))
        capture.release()

    def save_team(self, frame, pet_names, status_names, frame_nb):
        if not hasattr(self, "whole_pet_imgs"):
            video_name = os.path.splitext(os.path.basename(self.video_file))[0]
            self._load_whole_pets()

        team_img = frame[self.COORDS["team"]]
        shape = (int(1.3*team_img.shape[0]), team_img.shape[1], 3)
        visu = 255 * np.ones(shape, np.uint8)
        for i in range(5):
            if pet_names[i] is None:
                continue
            pet = self.whole_pet_imgs[pet_names[i]]
            if status_names[i] != 'Nothing':
                status = self.status[status_names[i]]

            h, w = pet.shape[:2]
            yt, xl = 5, 15 + 120*i
            yb, xr = yt + h, xl + w
            visu[yt:yb, xl:xr] = np.maximum(pet.img, 255 * (1 - pet.mask)[..., np.newaxis])

            if status_names[i] != 'Nothing':
                status.resize((35, 35))
                h, w = status.img.shape[:2]
                yt, xl = yb + 15, 40 + 120*i
                yb, xr = yt + h, xl + w
                visu[yt:yb, xl:xr] = np.maximum(status.img, 255 * (1 - status.mask)[..., np.newaxis])

        fig = plt.figure("main")
        axes = fig.subplots(2)
        axes[0].imshow(team_img)
        axes[1].imshow(visu)

        axes[0].axis('off')
        axes[1].axis('off')
        fig.savefig(os.path.join(self.output_path, f"team_{frame_nb}.png"))

    def extract_teams(self, nb_workers):
        self.logger.info("[EXTRACTOR] Initializing")
        workers_done = []
        frame, frame_nb = self.queue.get()
        while type(frame) is int:
            workers_done.append(frame)
            frame, frame_nb = self.queue.get()

        while len(workers_done) < nb_workers:
            self.logger.info("[EXTRACTOR] Processing frame")
            pets, status = self.extract_team(frame)
            self.save_team(frame, pets, status, frame_nb)

            self.logger.info("[EXTRACTOR] List of pets:" + str(pets))
            self.logger.info("[EXTRACTOR] List of status:" + str(status))
            frame, frame_nb = self.queue.get()
            while type(frame) is int:
                self.logger.info(f"[EXTRACTOR] Worker {frame} done")
                workers_done.append(frame)
                if len(workers_done) < nb_workers:
                    frame, frame_nb = self.queue.get()
                else:
                    break

        self.logger.info("[EXTRACTOR] Extractor done !")

    def run_sync(self):
        capture = cv2.VideoCapture(self.video_file)
        frame, frame_nb = self.goto_next_battle(capture)
        while frame is not None:
            self.logger.info(f"Frame {frame_nb}: battle found")
            pets, status = self.extract_team(frame)
            self.save_team(frame, pets, status, frame_nb)

            self.goto_next_turn(capture)
            frame, frame_nb = self.goto_next_battle(capture)

        capture.release()

    def run(self, nb_workers=2):
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
    if args.output is None:
        args.output = os.path.dirname(args.path)
    team_extractor = TeamExtractor(args.path, args.output)
    team_extractor.run(nb_workers=args.nb_workers)

