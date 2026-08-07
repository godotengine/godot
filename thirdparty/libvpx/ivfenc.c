/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./ivfenc.h"

#include "vpx/vpx_encoder.h"
#include "vpx_ports/mem_ops.h"

void ivf_write_file_header_with_video_info(FILE *outfile, unsigned int fourcc,
                                           int frame_cnt, int frame_width,
                                           int frame_height,
                                           vpx_rational_t timebase) {
  char header[32];

  header[0] = 'D';
  header[1] = 'K';
  header[2] = 'I';
  header[3] = 'F';
  mem_put_le16(header + 4, 0);              // version
  mem_put_le16(header + 6, 32);             // header size
  mem_put_le32(header + 8, fourcc);         // fourcc
  mem_put_le16(header + 12, frame_width);   // width
  mem_put_le16(header + 14, frame_height);  // height
  mem_put_le32(header + 16, timebase.den);  // rate
  mem_put_le32(header + 20, timebase.num);  // scale
  mem_put_le32(header + 24, frame_cnt);     // length
  mem_put_le32(header + 28, 0);             // unused

  fwrite(header, 1, 32, outfile);
}

void ivf_write_file_header(FILE *outfile, const struct vpx_codec_enc_cfg *cfg,
                           unsigned int fourcc, int frame_cnt) {
  ivf_write_file_header_with_video_info(outfile, fourcc, frame_cnt, cfg->g_w,
                                        cfg->g_h, cfg->g_timebase);
}

void ivf_write_frame_header(FILE *outfile, int64_t pts, size_t frame_size) {
  char header[12];

  mem_put_le32(header, (int)frame_size);
  mem_put_le32(header + 4, (int)(pts & 0xFFFFFFFF));
  mem_put_le32(header + 8, (int)(pts >> 32));
  fwrite(header, 1, 12, outfile);
}

void ivf_write_frame_size(FILE *outfile, size_t frame_size) {
  char header[4];

  mem_put_le32(header, (int)frame_size);
  fwrite(header, 1, 4, outfile);
}
