
# SuperAutoPets Team Extractor

This repository contains code that extracts enemy teams in the [SuperAutoPets](https://store.steampowered.com/app/1714040/Super_Auto_Pets/) game from Youtube videos of streamer.


## Usage

The extractor code needs the pets and objects images of SuperAutoPets, which are based on emojis from three opensource repositories:
- [Twitter emojis](https://github.com/twitter/twemoji)
- [Google emojis](https://github.com/googlefonts/noto-emoji)
- [Mozilla emojis](https://github.com/mozilla/fxemoji)

In order to download the necessary images, clone these three repositories in a directory (e.g. `emoji_dir`) and run the python script:
`python3 download_imgs.py path/to/emoji_dir/`

Once the images have been downloaded, run `python3 team_extractor.py path/to/video/` to start extracting every enemy team from a SuperAutoPets game video.

The script accept the option `--nb_workers` (`-n` in short form) to specify the number of threads to run in parallel to extract the teams.
If `-n 10` is used, the video is split in 10 equal parts, with one process on each part looking for battles, and an additional common process that extract teams from the frames corresponding to battles found by the other processes.

Outputs are yet to be defined.


## Requirements

- python 3
  + numpy
  + matplotlib
  + opencv-python
- inkscape (for the image downloader script, to convert SVG files to PNG)

