#!/usr/bin/env python

import argparse
import difflib
import glob
import os
import re
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+', help='Paths to directories containing checks')
    parser.add_argument('-f', '--filter', type=str, help='Display all team containing the given string')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-d', '--diff', action='store_true', help='Display differences between two directories')
    group.add_argument('-t', '--turns', action='store_true', help='Whether to validate turns')
    args = parser.parse_args()

    if args.diff and len(args.paths) < 2:
        parser.print_help()
        exit(1)

    return args


def get_all_files(path):
    files = []
    for (dirpath, _, filenames) in os.walk(path):
        if 'team_list.txt' in filenames:
            files.append(os.path.join(dirpath, 'team_list.txt'))
    return files


def check_turns(file):
    with open(file, 'r') as f:
        data = f.read().splitlines()
    turns = [int(line.split(' ', 1)[0]) for line in data if line]

    valid = True
    for i in range(len(turns)-1):
        if not (turns[i+1] == 1 or turns[i+1] == turns[i] + 1):
            valid = False

    if not valid:
        print(os.path.basename(os.path.dirname(file)))


def disp_teams(file, filter_str):
    with open(file, 'r') as f:
        teams = f.read().splitlines()

    team_imgs = glob.glob(os.path.join(os.path.dirname(file), "team_*.png"))
    team_imgs.sort(key=lambda file: int(os.path.basename(file)[5:-4]))

    for team, img in zip(teams, team_imgs):
        if not re.search(filter_str, team, re.I):
            continue

        print(img.ljust(50), team)
        open_cmd = ["eog", img]
        subprocess.run(open_cmd)


def get_diff(path1, path2, filter_str):
    dirs = os.listdir(path1)
    differ = difflib.Differ()
    for directory in dirs:
        print("Video", directory)

        dir_path1 = os.path.join(path1, directory)
        dir_path2 = os.path.join(path2, directory)
        if not os.path.isdir(dir_path2):
            print(f"\033[0;31mDirectory {directory} does not exist in {dir_path2}\033[0m")
            continue

        file1 = os.path.join(dir_path1, "team_list.txt")
        file2 = os.path.join(dir_path2, "team_list.txt")
        if not os.path.isfile(file1):
            print(f"\033[0;31mNo team_list.txt file in {os.path.join(dir_path1)}\033[0m")
            continue
        if not os.path.isfile(file2):
            print(f"\033[0;31mNo team_list.txt file in {os.path.join(dir_path2)}\033[0m")
            continue

        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            teams1 = f1.read().splitlines()
            teams2 = f2.read().splitlines()

        diff = list(differ.compare(teams1, teams2))
        i = 0
        changes = []
        while i < len(diff):
            if diff[i].startswith('- '):
                lines = [diff[i]]
                saved_ind = i
                i += 1
                while i < len(diff) and (diff[i].startswith('+ ') or diff[i].startswith('? ')):
                    lines.append(diff[i].replace('\n', ''))
                    diff.pop(i)
                changes.append((saved_ind, lines))
            else:
                i += 1

        new_imgs = glob.glob(os.path.join(dir_path2, "team_*.png"))
        new_imgs.sort(key=lambda file: int(os.path.basename(file)[5:-4]))

        for index, lines in changes:
            if filter_str and not re.search(filter_str, ' '.join(lines), re.I):
                continue

            print('\n'.join(lines))
            print()
            open_cmd = ["eog", new_imgs[index]]
            subprocess.run(open_cmd)


if __name__ == '__main__':
    args = parse_args()
    if args.turns:
        for path in args.paths:
            files = get_all_files(path)
            for file in files:
                check_turns(file)
    elif args.diff:
        get_diff(*args.paths, filter_str=args.filter)

    elif args.filter:
        for path in args.paths:
            files = get_all_files(path)
            for file in files:
                disp_teams(file, args.filter)
