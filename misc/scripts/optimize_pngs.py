#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Requires ImageMagic 7+
# Requires oxipng 9.1.3+

import multiprocessing
import subprocess
import sys


def quantize_image(filename, desired_colors):
    """
    Godot head have only 4 colors (counting transparent) but in non quantized form our
    files use dozens of colors. Usually 500 and up to 1000 in one of them. All additional
    colors are colors used to anti-alias the image. Lots of them are used in file only once
    or twice. They are mostly indistingusahble between each other so they are safe to remove
    When we reduce count of colors used by PNG file to 256, we can change it is mode to
    "Indexed". In this mode palette is created (up to 256 colors) and all the pixels are
    described not by using RGB(A) 3/4 bytes but by in index in palette 1 byte
    """

    # Retrieve amount of colors in the source image
    command = ["magick", filename, "-format", "%k", "info:-"]
    colors_count = int(subprocess.check_output(command))

    # Reduce amount of colors in image 1 by 1 for each step
    # It is much slower then reducing to desired amount of colors in one call
    # and gives slightly worse results but produces output without any artifacts
    for i in range(colors_count, desired_colors, -1):
        command = ["magick", filename, "-dither", "None", "-colors", str(i), filename]
        subprocess.run(command)


def optimize_image(filename):
    """
    Uses `oxipng's` "max" compression to compress image even further
    """
    command = ["oxipng", "-t1", "-omax", "-q", "--zopfli", "--zi", "255", "--strip", "all", filename]
    subprocess.run(command)


def process_image(filename):
    print(f"quantize_image: {filename}")
    quantize_image(filename, 255)
    print(f"optimize_image: {filename}")
    optimize_image(filename)


for i in range(1, len(sys.argv)):
    process = multiprocessing.Process(target=process_image, args=[sys.argv[i]])
    process.start()
