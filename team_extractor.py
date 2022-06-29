#!/usr/bin/env python

import argparse
import logging
import multiprocessing
import os

import cv2
import matplotlib.pyplot as plt
import matplotlib.widgets as wdg
import numpy as np

try:
    from PIL import Image
except ModuleNotFoundError:
    pass


PET_SIZE = 100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Path to the video to process.")
    parser.add_argument('-o', '--output', type=str, help="Dir in which to save output. Default to the path of the video to process.")
    parser.add_argument('-f', '--nb_finders', type=int, default=2, help="Number of battle finders to run in parallel.")
    parser.add_argument('-e', '--nb_extractors', type=int, default=2, help="Number of team extractors to run in parallel.")
    parser.add_argument('--sync', action='store_true', help="Extract without using multiple processes.")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.dirname(args.path)
    if args.sync:
        args.nb_finders = 1
        args.nb_extractors = 1
    if os.path.isdir(args.path) and 'video.mp4' in os.listdir(args.path):
        args.path = os.path.join(args.path, 'video.mp4')

    return args

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

    autoplay = autoplay[10:-10, 10:-10]
    mask = mask[10:-10, 10:-10]

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

def save_xp_bar(frame, spot, value):
    xp_bar = frame[TeamExtractor.COORDS["xp_bars"][spot]]
    os.makedirs("assets", exist_ok=True)
    img = Image.fromarray(xp_bar)
    img.save(f"assets/XP/xp_bar_{value}.png")

def save_xp_digit(frame, spot, digit):
    xp_digit = frame[TeamExtractor.COORDS["xp_digits"][spot]]
    os.makedirs("assets", exist_ok=True)
    img = Image.fromarray(xp_digit)
    img.save(f"assets/XP/xp_digit_{digit}.png")

def save_attack_digit(frame, spot, digit):
    # The digit 0 never lives alone -> take coords on an attack value of "10"
    digit_spot = (slice(8, 40), slice(14, 37) if digit > 0 else slice(20, 43))
    attack_digit_color = frame[TeamExtractor.COORDS["attacks"][spot]][digit_spot]
    attack_digit = cv2.cvtColor(attack_digit_color, cv2.COLOR_RGB2GRAY)

    h, w = attack_digit.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    seed = (15, 15) if digit % 7 else (12, 5)
    floodflags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    _, _, mask, _ = cv2.floodFill(attack_digit, mask, seed, (255, 0, 0), (25,)*3, (25,)*3, floodflags)

    mask = mask[1:-1, 1:-1]
    mask = cv2.dilate(mask, None)
    mask = cv2.dilate(mask, None)

    output = np.append(attack_digit_color, mask[..., np.newaxis], axis=2)
    os.makedirs("assets/stats", exist_ok=True)
    img = Image.fromarray(output)
    img.save(f"assets/stats/stat_{digit}.png")

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
    def __init__(self, img, mask, trim=False):
        self.curr_shape = img.shape[:2]
        self.init_img = img
        self.init_mask = mask

        if trim:
            img, mask = self.trim(img, mask)
        self.resized_imgs = {self.curr_shape: img}
        self.resized_masks = {self.curr_shape: mask}
        self.contours = {}

    @property
    def img(self):
        return self.resized_imgs[self.curr_shape]

    @property
    def mask(self):
        return self.resized_masks[self.curr_shape]

    @property
    def shape(self):
        return self.resized_imgs[self.curr_shape].shape

    def trim(self, img, mask):
        xs, ys = np.where(mask)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        return img[x_min:x_max+1, y_min:y_max+1], mask[x_min:x_max+1, y_min:y_max+1]

    def resize(self, new_shape):
        if new_shape not in self.resized_imgs:
            resized_img = cv2.resize(self.init_img, new_shape)
            resized_mask = cv2.resize(self.init_mask, new_shape)
            self.resized_imgs[new_shape] = resized_img
            self.resized_masks[new_shape] = resized_mask

        self.curr_shape = new_shape

    def _compute_contours(self):
        h, w = self.img.shape[:2]
        contours = np.zeros((h+2, w+2), dtype=np.bool_)
        contours[1:h+1, 1:w+1] = cv2.Canny(self.img, 400, 800)
        orig_size = (contours != 0).sum()

        shifted_contours = [contours[1:h+1, 1:w+1], contours[:h, 1:w+1], contours[2:, 1:w+1],
                            contours[1:h+1, :w], contours[1:h+1, 2:]]
        contours = (sum(shifted_contours) > 0)

        return contours, orig_size

    def get_contours(self):
        if self.curr_shape not in self.contours:
            self.contours[self.curr_shape] = self._compute_contours()
        return self.contours[self.curr_shape]


class TeamExtractor:

    COORDS = {}
    COORDS["autoplay"] = (slice(58, 137), slice(674, 789))
    COORDS["hourglass"] = (slice(25, 56), slice(401, 423))
    COORDS["turn"] = (slice(20, 60), slice(425, 480))

    COORDS["autoplay_area"] = extend(COORDS["autoplay"], 15)
    COORDS["hourglass_area"] = extend(COORDS["hourglass"], 15)

    COORDS["loading_area"] = (slice(90, 155), slice(12, 74))

    COORDS["attacks"] = []
    COORDS["lives"] = []
    COORDS["inter"] = []
    COORDS["pets"] = []
    COORDS["xp_digits"] = []
    COORDS["xp_bars"] = []

    def set_coords(self, frame):
        w0 = frame.shape[1] // 2
        res = cv2.matchTemplate(frame[:, w0:, ...], self.lvl, cv2.TM_SQDIFF)

        _, _, (w, h), _ = cv2.minMaxLoc(res)
        w = w0 + w - (w // 120) * 120
        for i in range(5):
            self.COORDS["attacks"].append((slice(h+132, h+183), slice(w-69+120*i, w-13+120*i)))
            self.COORDS["lives"].append((slice(h+132, h+183), slice(w-8+120*i, w+43+120*i)))
            self.COORDS["inter"].append((slice(h+140, h+174), slice(w-21+120*i, w-4+120*i)))
            self.COORDS["pets"].append((slice(h+10, h+128), slice(w-78+120*i, w+55+120*i)))
            self.COORDS["xp_digits"].append((slice(h-15, h+24), slice(w+22+120*i, w+52+120*i)))
            self.COORDS["xp_bars"].append((slice(h+18, h+38), slice(w-5+120*i, w+53+120*i)))
        self.COORDS["team"] = (slice(h-25, h+190), slice(660, 1275))

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
        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler())

        self.team_reprs = multiprocessing.Queue()

    def _load_pets(self):
        self.pets = {}
        self.whole_pets = {}
        for file in os.listdir("imgs/pets"):
            pet_name = file[:-4]
            img = cv2.imread(f"imgs/pets/{file}", cv2.IMREAD_UNCHANGED)
            if img.dtype == 'uint16':
                img = (img // 256).astype(np.uint8)
            mask = (img[:, :, 3] > 0).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img *= mask[:, :, np.newaxis]

            img = cv2.resize(img, (PET_SIZE, PET_SIZE))
            mask = cv2.resize(mask, (PET_SIZE, PET_SIZE))
            self.whole_pets[pet_name] = ImgStruct(img.copy(), mask.copy())

            # Remove the head position from the mask to account for hats
            # Mosquito and Scorpions have their hat very low, remove the precise area
            if pet_name == "Mosquito":
                img[37:60, 20:55] = 0
                img[:20] = 0
                mask[37:60, 20:55] = 0
                mask[:20] = 0
            elif pet_name == "Scorpion":
                img[12:60, 35:70] = 0
                mask[12:60, 35:70] = 0
            else:
                # Other pets just have the hat on top of them: mask the first rows
                img = img[30:, :]
                mask = mask[30:, :]

            self.pets[pet_name] = ImgStruct(img, mask, trim=True)

    def _load_status(self):
        self.status = {}
        for file in os.listdir("imgs/status"):
            status_name = file[:-4]
            img = cv2.imread(f"imgs/status/{file}", cv2.IMREAD_UNCHANGED)
            if img.dtype == 'uint16':
                img = (img // 256).astype(np.uint8)
            mask = (img[:, :, 3] > 0).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img *= mask[:, :, np.newaxis]
            img = cv2.resize(img, (PET_SIZE // 2, PET_SIZE // 2))
            mask = cv2.resize(mask, (PET_SIZE // 2, PET_SIZE // 2))
            self.status[status_name] = ImgStruct(img, mask)

    def _load_assets(self):
        self.autoplay = cv2.imread("assets/autoplay_icon.png", cv2.IMREAD_UNCHANGED)
        self.autoplay_mask = self.autoplay[:, :, 3]
        self.autoplay = cv2.cvtColor(self.autoplay, cv2.COLOR_BGR2RGB)

        self.hourglass = cv2.imread("assets/hourglass_icon.png")
        self.hourglass = cv2.cvtColor(self.hourglass, cv2.COLOR_BGR2RGB)

        self.lvl = cv2.imread("assets/lvl_icon.png")
        self.lvl = cv2.cvtColor(self.lvl, cv2.COLOR_BGR2RGB)

        self.xp_digits = []
        for i in range(1, 4):
            xp_digit = cv2.imread(f"assets/XP/xp_digit_{i}.png")
            xp_digit = cv2.cvtColor(xp_digit, cv2.COLOR_BGR2RGB)
            self.xp_digits.append(xp_digit)

        self.xp_bars = []
        for i in range(5):
            xp_bar = cv2.imread(f"assets/XP/xp_bar_{i}.png")
            xp_bar = cv2.cvtColor(xp_bar, cv2.COLOR_BGR2RGB)
            self.xp_bars.append(xp_bar)

        self.xp_conversion_table = {(1, 0): 0, (1, 2): 1, (2, 0): 2,
                                    (2, 1): 3, (2, 3): 4, (3, 4): 5}

        self.stat_digits = []
        for i in range(10):
            stat_digit = cv2.imread(f"assets/digits/digit_{i}.png", cv2.IMREAD_UNCHANGED)
            stat_mask = stat_digit[:, :, 3] // 255
            stat_digit = cv2.cvtColor(stat_digit, cv2.COLOR_BGR2GRAY)
            self.stat_digits.append((stat_digit, stat_mask))

        self.turn_digits = []
        for i in range(10):
            turn_digit = cv2.resize(self.stat_digits[i][0], (18, 26))
            turn_mask = cv2.resize(self.stat_digits[i][1], (18, 26))
            self.turn_digits.append((turn_digit, turn_mask))

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
            inter = frame[self.COORDS["inter"][spot]]
            white_pixels = (inter.mean(axis=2) > 245).sum()

            if white_pixels >= 350:
                spots.append(spot)
        return spots

    def get_pet_score(self, frame, pet):
        res = cv2.matchTemplate(frame, pet.img, cv2.TM_SQDIFF, mask=pet.mask)
        _, _, loc, _ = cv2.minMaxLoc(res)
        found_img = frame[loc[1]:loc[1]+pet.shape[0], loc[0]:loc[0]+pet.shape[1]]

        close_pixels = (np.abs(found_img.astype(np.int16) - pet.img).mean(axis=2) < 12) * pet.mask
        score = 100 * close_pixels.sum() / pet.mask.sum()

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
                dh = (pet_area.shape[0] - pet.shape[0]) // 2
                dw = (pet_area.shape[1] - pet.shape[1]) // 4
                area = pet_area[dh:, dw:-dw]
                scores[pet_name] = self.get_pet_score(area, pet)

            team.append(max(scores, key=scores.get))

        return team

    def get_status_score(self, frame, status, shape):
        status.resize(shape)
        res = cv2.matchTemplate(frame, status.img, cv2.TM_SQDIFF, mask=status.mask)
        _, _, loc, _ = cv2.minMaxLoc(res)
        found_img = frame[loc[1]:loc[1]+status.shape[0], loc[0]:loc[0]+status.shape[1]]

        # Closeness score: nb of pixels whose RGB values are close to status img (to maximize)
        close_pixels = (np.abs(found_img.astype(np.int16) - status.img).mean(axis=2) < 15) * status.mask
        closeness_score = 100 * close_pixels.sum() / status.mask.sum()

        # Peak score: nb of location in the frame that results in almost the minimum of score
        # found in the convolution (to minimize)
        nb_peaks = (res < 1.2*res.min()).sum()

        # Nb of contours in common with the status img (to maximize)
        found_contours = cv2.Canny(found_img, 400, 800).view(np.bool_)
        contours, contours_size = status.get_contours()
        contours_score = 100 * (contours * found_contours).sum() / contours_size

        return closeness_score, nb_peaks, contours_score

    def extract_status(self, frame, spots):
        all_status = []
        for spot in range(5):
            if spot not in spots:
                all_status.append(None)
                continue

            pet_area = frame[self.COORDS["pets"][spot]]
            for status_name, status in self.status.items():
                possible_sizes = list(range(25, 50, 5)) + [28]
                for size in possible_sizes:
                    closeness_score, nb_peaks, contours_score = self.get_status_score(pet_area, status, (size, size))

                    score = (closeness_score - 36) + (20 - nb_peaks) + (contours_score - 60) // 2
                    if score > 30:
                        all_status.append(status_name)
                        break

                else:
                    continue
                break

            else:
                all_status.append("Nothing")

        return all_status

    def extract_xps(self, frame, spots):
        xps = []
        for spot in range(5):
            if spot not in spots:
                xps.append(None)
                continue

            xp_digit_area = frame[self.COORDS["xp_digits"][spot]]
            digit_scores = []
            for xp_digit in self.xp_digits:
                res = cv2.matchTemplate(xp_digit_area, xp_digit, cv2.TM_SQDIFF)
                digit_scores.append(res.min())
            xp_digit = min(range(3), key=lambda i: digit_scores[i]) + 1

            if xp_digit == 3:
                xps.append(5)
                continue

            xp_bar_area = frame[self.COORDS["xp_bars"][spot]]
            bar_scores = []
            for xp_bar in self.xp_bars:
                res = cv2.matchTemplate(xp_bar_area, xp_bar, cv2.TM_SQDIFF)
                bar_scores.append(res.min())
            xp_bar_value = min(range(5), key=lambda i: bar_scores[i])

            xps.append(self.xp_conversion_table[(xp_digit, xp_bar_value)])

        return xps

    def extract_digit(self, frame, digit_set, filter_one=False):
        found_digits = []
        for i in range(10):
            digit, mask = digit_set[i]
            res = 1000 * cv2.matchTemplate(frame, digit, cv2.TM_SQDIFF_NORMED, mask=mask)

            if res.min() < 100:
                _, xs = np.where(res <= max(10, 3*res.min()))

                # Clustering: locations should be more than 5 pixels apart
                xs.sort()
                prev_x = -100
                for x in xs:
                    if x - prev_x > 5:
                        found_digits.append((x, i))
                    prev_x = x

        found_digits.sort()

        # Filter out 1 after 4 because it's basically impossible and the right
        # section of the 4 looks like a 1
        if filter_one:
            for i in range(len(found_digits)-2, -1, -1):
                close = (abs(found_digits[i][0] - found_digits[i+1][0]) < 5)
                if close and found_digits[i][1] == 4 and found_digits[i+1][1] == 1:
                    found_digits.pop(i+1)

        value = 0
        for _, digit in found_digits:
            value = 10 * value + digit

        return value

    def extract_stats(self, frame, spots):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        stats = []
        for spot in range(5):
            if spot not in spots:
                stats.append(None)
                continue

            attack_area = frame[self.COORDS["attacks"][spot]]
            life_area = frame[self.COORDS["lives"][spot]]

            attack = self.extract_digit(attack_area, self.stat_digits)
            life = self.extract_digit(life_area, self.stat_digits)
            stats.append((attack, life))

        return stats

    def extract_team(self, frame):
        if not self.COORDS["pets"]:
            self.set_coords(frame)

        spots = self.find_spots(frame)
        pets = self.extract_pets(frame, spots)
        status = self.extract_status(frame, spots)
        xps = self.extract_xps(frame, spots)
        stats = self.extract_stats(frame, spots)
        return pets, status, xps, stats

    def extract_turn(self, frame):
        turn_area = cv2.cvtColor(frame[self.COORDS["turn"]], cv2.COLOR_RGB2GRAY)
        turn = self.extract_digit(turn_area, self.turn_digits, True)

        # The very first frames when running the game start at 0
        if turn == 0:
            turn = 1
        return turn

    def goto_next(self, capture, coords, img, mask=None, skip_loading=False):
        frame_nb = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        skip = 40
        frame = self.get_frame(capture)
        while True:
            if frame is None:
                return None, -1

            area = frame[coords]
            res = cv2.matchTemplate(area, img, cv2.TM_SQDIFF, mask=mask)
            _, _, loc, _ = cv2.minMaxLoc(res)
            found_img = area[loc[1]:loc[1]+img.shape[0], loc[0]:loc[0]+img.shape[1]]

            close_pixels = (np.abs(found_img.astype(np.int16) - img).mean(axis=2) < 25)
            if mask is not None:
                close_pixels *= mask.astype('bool')
                closeness_score = 100 * close_pixels.sum() / (mask != 0).sum()
                contours = cv2.Canny(area, 100, 200) // 255
                found = (closeness_score > 50) and (contours.sum() > 800)
            else:
                closeness_score = 100 * close_pixels.sum() / close_pixels.size
                found = (closeness_score > 80)

            if skip_loading and found:
                loading_area = frame[self.COORDS["loading_area"]]
                white_pixels = (loading_area.mean(axis=2) > 245).sum()
                if white_pixels > 800:
                    found = False

            if found:
                if skip > 1:
                    frame_nb -= skip
                    skip = (1 if skip < 10 else skip // 2)
                else:
                    break

            frame_nb += skip
            frame = self.get_frame(capture, frame_nb)

        return frame, frame_nb + 1

    def goto_next_battle(self, capture):
        return self.goto_next(capture, self.COORDS["autoplay_area"], self.autoplay, self.autoplay_mask)

    def goto_next_turn(self, capture):
        frame, frame_nb = self.goto_next(capture, self.COORDS["hourglass_area"], self.hourglass, skip_loading=True)
        if frame is None:
            return None, -1
        return self.extract_turn(frame), frame_nb

    def find_battles(self, worker_id, init_frame, end_frame):
        self.logger.info(f"[WORKER {worker_id}] Running between frames {init_frame} and {end_frame}")
        capture = cv2.VideoCapture(self.video_file)
        capture.set(cv2.CAP_PROP_POS_FRAMES, init_frame)

        turn, _ = self.goto_next_turn(capture)
        frame, frame_nb = self.goto_next_battle(capture)
        while frame is not None and frame_nb < end_frame:
            self.logger.info(f"[WORKER {worker_id}] Battle found at frame {frame_nb} ! Putting in queue")
            self.queue.put((frame, frame_nb, turn))
            turn, turn_frame_nb = self.goto_next_turn(capture)
            frame, frame_nb = self.goto_next_battle(capture)

        # Process battles after the end that could not be computed by next worker
        # (because of turn_frame before its starting frame)
        if frame is not None and turn_frame_nb < end_frame <= frame_nb:
            capture.set(cv2.CAP_PROP_POS_FRAMES, end_frame)
            _, new_turn_frame_nb = self.goto_next_turn(capture)
            if new_turn_frame_nb > frame_nb:
                self.queue.put((frame, frame_nb, turn))

        self.logger.info(f"[WORKER {worker_id}] Done !")
        capture.release()

    def save_team_img(self, frame, turn, pet_names, status_names, xps, stats, frame_nb):
        team_img = frame[self.COORDS["team"]]
        shape = (int(1.8*team_img.shape[0]), team_img.shape[1], 3)

        visu_turn = 255 * np.ones((shape[0], 100, 3), np.uint8)
        h, w = self.hourglass.shape[:2]
        visu_turn[180:180+h, 5:5+w] = self.hourglass
        x = 45
        for i in map(int, str(turn)):
            digit, mask = self.turn_digits[i]
            digit = digit[..., np.newaxis].repeat(3, axis=2)
            h, w = digit.shape[:2]
            visu_turn[187:187+h, x:x+w] = np.maximum(digit, 255 * (1 - mask)[..., np.newaxis])
            x += 25

        visu = 255 * np.ones(shape, np.uint8)
        for i in range(5):
            if pet_names[i] is None:
                continue
            pet = self.whole_pets[pet_names[i]]

            h, w = pet.shape[:2]
            y, x = 2, 15 + 120*i
            visu[y:y+h, x:x+w] = np.maximum(pet.img, 255 * (1 - pet.mask)[..., np.newaxis])

            y, x = y + 110, 43 + 120*i
            if status_names[i] != 'Nothing':
                status = self.status[status_names[i]]
                status.resize((35, 35))
                h, w = status.shape[:2]
                visu[y:y+h, x:x+w] = np.maximum(status.img, 255 * (1 - status.mask)[..., np.newaxis])

            for key, value in self.xp_conversion_table.items():
                if value == xps[i]:
                    digit, bar = key
                    break
            xp_digit = self.xp_digits[digit - 1]
            h, w = xp_digit.shape[:2]
            y, x = y + 50, 55 + 120*i
            visu[y:y+h, x:x+w] = xp_digit

            xp_bar = self.xp_bars[bar]
            h, w = xp_bar.shape[:2]
            y, x = y + 50, 40 + 120*i
            visu[y:y+h, x:x+w] = xp_bar

            for j in range(2):
                stat = stats[i][j]
                stat_digits = list(map(int, str(stat)))
                y, x = y + 50, 40 + 120*i
                for digit in stat_digits:
                    stat_digit, stat_mask = self.stat_digits[digit]
                    stat_digit = stat_digit * stat_mask + 255 * (1 - stat_mask)
                    h, w = stat_digit.shape[:2]
                    visu[y:y+h, x:x+w] = cv2.cvtColor(stat_digit, cv2.COLOR_GRAY2RGB)
                    x += 22

        visu = np.hstack((visu_turn, visu))
        fig = plt.figure("main")
        axes = fig.subplots(2)
        axes[0].imshow(team_img)
        axes[1].imshow(visu)

        axes[0].axis('off')
        axes[1].axis('off')
        fig.savefig(os.path.join(self.output_path, f"team_{frame_nb}.png"))

    def save_team(self, frame, turn, pet_names, status_names, xps, stats, frame_nb):
        self.save_team_img(frame, turn, pet_names, status_names, xps, stats, frame_nb)
        pets_reprs = []
        for i in range(5):
            if pet_names[i] is None:
                continue

            pet_name = pet_names[i].replace(' ', '_').lower()
            if status_names[i] == "Nothing":
                status_name = "none"
            else:
                status_name = status_names[i].replace(' ', '_').lower().replace('_alt', '')
            attack, life = stats[i]
            pets_reprs.append(f"({pet_name} {attack} {life} {xps[i]} {status_name})")

        self.team_reprs.put((frame_nb, turn, ' '.join(pets_reprs)))

    def remove_replays(self, teams):
        new_teams = [teams[0]]
        for team in teams:
            if team[2] != new_teams[-1][2]:
                new_teams.append(team)
        return new_teams

    def write_teams(self):
        team_file = os.path.join(self.output_path, "team_list.txt")
        if os.path.isfile(team_file):
            os.rename(team_file, team_file + ".old")
        self.logger.info(f"[WRITER]: Writing extracted teams to {team_file}")
        teams = []
        while not self.team_reprs.empty():
            teams.append(self.team_reprs.get())

        teams.sort()
        teams = self.remove_replays(teams)
        teams_str = '\n'.join([f"{turn} {team_str}" for _, turn, team_str in teams])

        with open(team_file, 'w') as file:
            file.write(teams_str)

    def extract_teams(self, extractor_id):
        self.logger.info(f"[EXTRACTOR {extractor_id}] Initializing")
        frame, frame_nb, turn = self.queue.get()

        while frame is not None:
            self.logger.info(f"[EXTRACTOR {extractor_id}] Processing frame {frame_nb}")
            pets, status, xps, stats = self.extract_team(frame)
            self.save_team(frame, turn, pets, status, xps, stats, frame_nb)

            self.logger.info(f"[EXTRACTOR {extractor_id}] List of pets: {pets}")
            self.logger.info(f"[EXTRACTOR {extractor_id}] List of status: {status}")
            frame, frame_nb, turn = self.queue.get()

        self.logger.info(f"[EXTRACTOR {extractor_id}] Extractor done !")

    def run_sync(self):
        capture = cv2.VideoCapture(self.video_file)
        turn, _ = self.goto_next_turn(capture)
        frame, frame_nb = self.goto_next_battle(capture)
        while frame is not None:
            self.logger.info(f"Frame {frame_nb}: battle found")
            pets, status, xps, stats = self.extract_team(frame)
            self.save_team(frame, turn, pets, status, xps, stats, frame_nb)

            turn, _ = self.goto_next_turn(capture)
            frame, frame_nb = self.goto_next_battle(capture)

        self.write_teams()
        capture.release()

    def run(self, nb_finders=2, nb_extractors=2):
        if nb_finders == nb_extractors == 1:
            self.run_sync()
            return

        frame_limits = [(i*self.video_length) // nb_finders for i in range(nb_finders+1)]

        battle_finders = []
        for i in range(nb_finders):
            battle_finder = multiprocessing.Process(target=self.find_battles, args=(i, *frame_limits[i:i+2]))
            battle_finders.append(battle_finder)

        team_extractors = []
        for i in range(nb_extractors):
            team_extractor = multiprocessing.Process(target=self.extract_teams, args=(i, ))
            team_extractors.append(team_extractor)

        for battle_finder in battle_finders:
            battle_finder.start()
        for team_extractor in team_extractors:
            team_extractor.start()

        for battle_finder in battle_finders:
            battle_finder.join()
        for _ in range(nb_extractors):
            self.queue.put((None, -1, -1))
        for team_extractor in team_extractors:
            team_extractor.join()

        self.write_teams()


if __name__ == '__main__':
    args = parse_args()
    team_extractor = TeamExtractor(args.path, args.output)
    team_extractor.run(nb_finders=args.nb_finders, nb_extractors=args.nb_extractors)
