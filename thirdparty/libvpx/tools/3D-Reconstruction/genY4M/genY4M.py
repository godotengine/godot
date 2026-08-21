##  Copyright (c) 2020 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##

import argparse
from os import listdir, path
from PIL import Image
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--frame_path", default="../data/frame/", type=str)
parser.add_argument("--frame_rate", default="25:1", type=str)
parser.add_argument("--interlacing", default="Ip", type=str)
parser.add_argument("--pix_ratio", default="0:0", type=str)
parser.add_argument("--color_space", default="4:2:0", type=str)
parser.add_argument("--output", default="output.y4m", type=str)


def generate(args, frames):
  if len(frames) == 0:
    return
  #sort the frames based on the frame index
  frames = sorted(frames, key=lambda x: x[0])
  #convert the frames to YUV form
  frames = [f.convert("YCbCr") for _, f in frames]
  #write the header
  header = "YUV4MPEG2 W%d H%d F%s %s A%s" % (frames[0].width, frames[0].height,
                                             args.frame_rate, args.interlacing,
                                             args.pix_ratio)
  cs = args.color_space.split(":")
  header += " C%s%s%s\n" % (cs[0], cs[1], cs[2])
  #estimate the sample step based on subsample value
  subsamples = [int(c) for c in cs]
  r_step = [1, int(subsamples[2] == 0) + 1, int(subsamples[2] == 0) + 1]
  c_step = [1, 4 // subsamples[1], 4 // subsamples[1]]
  #write in frames
  with open(args.output, "wb") as y4m:
    y4m.write(header)
    for f in frames:
      y4m.write("FRAME\n")
      px = f.load()
      for k in xrange(3):
        for i in xrange(0, f.height, r_step[k]):
          for j in xrange(0, f.width, c_step[k]):
            yuv = px[j, i]
            y4m.write(chr(yuv[k]))


if __name__ == "__main__":
  args = parser.parse_args()
  frames = []
  frames_mv = []
  for filename in listdir(args.frame_path):
    name, ext = filename.split(".")
    if ext == "png":
      name_parse = name.split("_")
      idx = int(name_parse[-1])
      img = Image.open(path.join(args.frame_path, filename))
      if name_parse[-2] == "mv":
        frames_mv.append((idx, img))
      else:
        frames.append((idx, img))
  if len(frames) == 0:
    print("No frames in directory: " + args.frame_path)
    sys.exit()
  print("----------------------Y4M Info----------------------")
  print("width:  %d" % frames[0][1].width)
  print("height: %d" % frames[0][1].height)
  print("#frame: %d" % len(frames))
  print("frame rate: %s" % args.frame_rate)
  print("interlacing: %s" % args.interlacing)
  print("pixel ratio: %s" % args.pix_ratio)
  print("color space: %s" % args.color_space)
  print("----------------------------------------------------")

  print("Generating ...")
  generate(args, frames)
  if len(frames_mv) != 0:
    args.output = args.output.replace(".y4m", "_mv.y4m")
    generate(args, frames_mv)
