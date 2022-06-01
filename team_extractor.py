#!/usr/bin/env python

import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


PET_SIZE = 100


def extend(coords, dx, dy=None):
    if dy is None:
        dy = dx
    x, y = coords
    return (slice(x.start-dx, x.stop+dx), slice(y.start-dy, y.stop+dy))

# COORDS_AUTOPLAY = (slice(20, 77), slice(329, 403))
# COORDS_HOURGLASS = (slice(5, 35), slice(190, 220))

COORDS_AUTOPLAY = (slice(58, 137), slice(674, 789))
COORDS_AUTOPLAY_AREA = extend(COORDS_AUTOPLAY, 15)

COORDS_HOURGLASS = (slice(25, 56), slice(401, 423))
COORDS_HOURGLASS_AREA = extend(COORDS_HOURGLASS, 15)

COORDS_TEAM = (slice(430, 600), slice(660, 1275))
COORDS_ATTACK = [(slice(546, 595), slice(670+120*i, 719+120*i)) for i in range(5)]
COORDS_LIFE = [(slice(546, 595), slice(729+120*i, 778+120*i)) for i in range(5)]


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
    _, _, loc, _ = cv2.minMaxLoc(res)
    found_img = frame[loc[1]:loc[1]+img.shape[0], loc[0]:loc[0]+img.shape[1]]

    close_pixels = (np.abs(found_img.astype(np.int16) - img).mean(axis=2) < 10)
    score = 100 * close_pixels.sum() / mask.size
    return score


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

    def get_frame(self, frame_id=None):
        if frame_id is not None:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self.video.read()
        if not ret:
            raise Exception("Couldn't read frame")

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
            # show(frame[COORDS_ATTACK[spot]])
            # breakpoint()
            # show(cv2.Canny(frame[COORDS_ATTACK[spot]], 800, 1200))
        return spots

    def extract_status(self, frame, spots):
        all_status = []
        for spot in spots:
            xl, xr = COORDS_ATTACK[spot][1].start - 8, COORDS_LIFE[spot][1].stop + 8
            yt, yb = COORDS_ATTACK[spot][0].start - 130, COORDS_ATTACK[spot][0].start - 3
            pet_area = frame[yt:yb, xl:xr]

            # min_all_status = 1000000
            for status_name, (status_img, status_mask) in self.status_imgs.items():
                # min_status = 1000000
                # min_size = 0
                for size in range(30, 50, 5):
                    resized_status_img = cv2.resize(status_img, (size, size))
                    resized_status_mask = (resized_status_img.sum(axis=2) != 0).astype(np.uint8)

                    score = get_found_score(pet_area, resized_status_img, resized_status_mask)
                    if score > 30:
                        all_status.append(status_name)
                        break

                else:
                    continue
                break

            else:
                all_status.append(None)


#                     status_res = cv2.matchTemplate(pet_area, resized_status_img, cv2.TM_SQDIFF,
#                                                    mask=resized_status_mask)
#                     found = (status_res <= 1.2*status_res.min()).sum()
#                     if found < min_status:
#                         min_status = found
#                         min_size = size

#                 if min_status <= min_all_status:
#                     min_all_status = min_status
#                     status_found = status_name
                    # size_found = min_size

            # if spot == 3:
            #     breakpoint()

            # if min_all_status <= 5:
                # print(f"Found {status_found} on spot {spot}")
                # all_status.append(status_found)
            # else:
                # all_status.append(None)

                # status_img, status_mask = self.status_imgs[status_found]
                # resized_status_img = cv2.resize(status_img, (size_found, size_found))
                # resized_status_mask = (resized_status_img.sum(axis=2) != 0).astype(np.uint8)
                # status_res = cv2.matchTemplate(pet_area, resized_status_img, cv2.TM_SQDIFF,
                #                                mask=resized_status_mask)
                # threshold = 1.2*status_res.min()
                # loc = np.where(status_res < threshold)

                # h, w = resized_status_img.shape[:2]
                # rectangles = [(x, y, x+w, y+h) for (y, x) in zip(*loc)]
                # rectangles *= 2
                # rectangles, _ = cv2.groupRectangles(rectangles, 1)

                # all_status_masks[spot] = (resized_status_mask, rectangles)

                # img = pet_area.copy()
                # for (x, y, xw, yh) in rectangles:
                #     cv2.rectangle(img, (x, y), (xw, yh), (0, 255, 0), 2)
                # print([rect[0] for rect in rectangles])
                # plt.imshow(img)
                # plt.show()

        return all_status

    # def extract_pet_from_frame(self, frame, pet_name):
    #     pet_img, pet_mask = self.pet_imgs[pet_name]
    #     res = cv2.matchTemplate(frame, pet_img, cv2.TM_SQDIFF, mask=pet_mask)
    #     _, _, loc, _ = cv2.minMaxLoc(res)
    #     found_pet = frame[loc[1]:loc[1]+pet_img.shape[0], loc[0]:loc[0]+pet_img.shape[1]]

    #     close_pixels = (np.abs(found_pet.astype(np.int16) - pet_img).mean(axis=2) < 10)
    #     percentage = close_pixels.sum() / pet_mask.size
    #     return percentage

    def extract_pets(self, frame, spots, status):
        team = []
        for spot in spots:
            xl, xr = COORDS_ATTACK[spot][1].start - 8, COORDS_LIFE[spot][1].stop + 8
            yt, yb = COORDS_ATTACK[spot][0].start - 130, COORDS_ATTACK[spot][0].start - 3
            pet_area = frame[yt:yb, xl:xr]

            scores = {}
            # test = {}
            # yo = {}
            for pet_name, (pet_img, pet_mask) in self.pet_imgs.items():
                scores[pet_name] = get_found_score(pet_area, pet_img, pet_mask)
                # res = cv2.matchTemplate(pet_area, pet_img, cv2.TM_SQDIFF, mask=pet_mask)
                # scores[pet_name] = (res <= 1.2*res.min()).sum()
                # test[pet_name] = res

                # _, _, loc, _ = cv2.minMaxLoc(res)
                # found_pet = pet_area[loc[1]:loc[1]+pet_img.shape[0], loc[0]:loc[0]+pet_img.shape[1]]

                # close_pixels = (np.abs(found_pet.astype(np.int16) - pet_img).mean(axis=2) < 10)
                # percentage = close_pixels.sum() / pet_mask.size
                # scores[pet_name] = percentage

            # print(sorted(scores.values(), reverse=True)[:10])
            team.append(max(scores, key=scores.get))
            # print("Spot", spot, min(scores, key=scores.get))
            # if spot == 3:
            #     _, _, loc, _ = cv2.minMaxLoc(test['Fish'])
            #     img = pet_area.copy()
            #     fish_img, fish_mask = self.pet_imgs['Fish']
            #     for i in range(fish_img.shape[0]):
            #         for j in range(fish_img.shape[1]):
            #             if fish_mask[i, j]:
            #                 img[i+loc[1], j+loc[0]] = fish_img[i, j]

            #     show(pet_area, img)
            #     breakpoint()
        return team

    def extract_team(self, frame):
        spots = self.find_spots(frame)
        status = self.extract_status(frame, spots)
        pets = self.extract_pets(frame, spots, status)
        return pets, status

    def _extract_pets(self, frame):
        team = []
        frame = frame[COORDS_TEAM]
        print("Extracting frame ", self.video.get(cv2.CAP_PROP_POS_FRAMES))
        for pet_name, (pet_img, pet_mask) in self.pet_imgs.items():
            res = cv2.matchTemplate(frame, pet_img, cv2.TM_SQDIFF, mask=pet_mask)

            if (res <= 1.2*res.min()).sum() >= 20:
                continue

            threshold = 1.2*res.min()
            loc = np.where(res < threshold)

            h, w = pet_img.shape[:2]
            rectangles = [(x, y, x+w, y+h) for (y, x) in zip(*loc)]
            rectangles *= 2
            rectangles, _ = cv2.groupRectangles(rectangles, 1)

            img = frame.copy()
            print(pet_name, len(rectangles))
            for (x, y, xw, yh) in rectangles:
                cv2.rectangle(img, (x, y), (xw, yh), (0, 255, 0), 2)
                # for i in range(x, xw):
                #     for j in range(y, yh):
                #         if pet_mask[j-y, i-x]:
                #             img[j, i] = pet_img[j-y, i-x]
            print([rect[0] for rect in rectangles])
            plt.imshow(img)
            plt.show()
            for (x, y, xw, yh) in rectangles:
                team.append((pet_name, x))

        team.sort(key=lambda p: p[1])
        team = list(map(lambda p: p[0], team))
        return team

    def goto_next_battle(self):
        print("Looking for battle")
        while True:
            frame = self.get_frame()
            res = cv2.matchTemplate(frame[COORDS_AUTOPLAY_AREA], self.autoplay, cv2.TM_SQDIFF, mask=self.autoplay_mask)
            if (res <= 1.2*res.min()).sum() <= 20:
                # print(self.video.get(cv2.CAP_PROP_POS_FRAMES))
                break
        return frame

    def goto_next_turn(self):
        print("Looking for new turn")
        while True:
            frame = self.get_frame()
            res = cv2.matchTemplate(frame[COORDS_HOURGLASS_AREA], self.hourglass, cv2.TM_SQDIFF)
            if (res <= 1.2*res.min()).sum() <= 20:
                self.get_frame()   # Wait one frame to pass black screen
                # show(frame)
                break

    def extract_all_teams(self):
        # self.video.set(cv2.CAP_PROP_POS_FRAMES, 9000)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 10500)
        frame = self.goto_next_battle()
        while frame is not None:
            print("Frame", self.video.get(cv2.CAP_PROP_POS_FRAMES))
            pets, status = self.extract_team(frame)
            print("List of pets:", pets)
            print("List of status:", status)
            show(frame)
            # breakpoint()
            self.goto_next_turn()
            frame = self.goto_next_battle()

        self.video.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Path to the video to process")
    args = parser.parse_args()

    team_extractor = TeamExtractor(args.path)
    team_extractor.extract_all_teams()

