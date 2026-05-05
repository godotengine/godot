/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <stdlib.h>
#include <string.h>

#include "./ivfdec.h"
#include "./video_reader.h"

#include "vpx_ports/mem_ops.h"

static const char *const kIVFSignature = "DKIF";

struct VpxVideoReaderStruct {
  VpxVideoInfo info;
  FILE *file;
  uint8_t *buffer;
  size_t buffer_size;
  size_t frame_size;
};

VpxVideoReader *vpx_video_reader_open(const char *filename) {
  char header[32];
  VpxVideoReader *reader = NULL;
  FILE *const file = fopen(filename, "rb");
  if (!file) {
    fprintf(stderr, "%s can't be opened.\n", filename);  // Can't open file
    return NULL;
  }

  if (fread(header, 1, 32, file) != 32) {
    fprintf(stderr, "File header on %s can't be read.\n",
            filename);  // Can't read file header
    return NULL;
  }
  if (memcmp(kIVFSignature, header, 4) != 0) {
    fprintf(stderr, "The IVF signature on %s is wrong.\n",
            filename);  // Wrong IVF signature

    return NULL;
  }
  if (mem_get_le16(header + 4) != 0) {
    fprintf(stderr, "%s uses the wrong IVF version.\n",
            filename);  // Wrong IVF version

    return NULL;
  }

  reader = calloc(1, sizeof(*reader));
  if (!reader) {
    fprintf(
        stderr,
        "Can't allocate VpxVideoReader\n");  // Can't allocate VpxVideoReader

    return NULL;
  }

  reader->file = file;
  reader->info.codec_fourcc = mem_get_le32(header + 8);
  reader->info.frame_width = mem_get_le16(header + 12);
  reader->info.frame_height = mem_get_le16(header + 14);
  reader->info.time_base.numerator = mem_get_le32(header + 16);
  reader->info.time_base.denominator = mem_get_le32(header + 20);

  return reader;
}

void vpx_video_reader_close(VpxVideoReader *reader) {
  if (reader) {
    fclose(reader->file);
    free(reader->buffer);
    free(reader);
  }
}

int vpx_video_reader_read_frame(VpxVideoReader *reader) {
  return !ivf_read_frame(reader->file, &reader->buffer, &reader->frame_size,
                         &reader->buffer_size);
}

const uint8_t *vpx_video_reader_get_frame(VpxVideoReader *reader,
                                          size_t *size) {
  if (size) *size = reader->frame_size;

  return reader->buffer;
}

const VpxVideoInfo *vpx_video_reader_get_info(VpxVideoReader *reader) {
  return &reader->info;
}
