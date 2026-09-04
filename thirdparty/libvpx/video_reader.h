/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VIDEO_READER_H_
#define VPX_VIDEO_READER_H_

#include "./video_common.h"

// The following code is work in progress. It is going to  support transparent
// reading of input files. Right now only IVF format is supported for
// simplicity. The main goal the API is to be simple and easy to use in example
// code and in vpxenc/vpxdec later. All low-level details like memory
// buffer management are hidden from API users.
struct VpxVideoReaderStruct;
typedef struct VpxVideoReaderStruct VpxVideoReader;

#ifdef __cplusplus
extern "C" {
#endif

// Opens the input file for reading and inspects it to determine file type.
// Returns an opaque VpxVideoReader* upon success, or NULL upon failure.
// Right now only IVF format is supported.
VpxVideoReader *vpx_video_reader_open(const char *filename);

// Frees all resources associated with VpxVideoReader* returned from
// vpx_video_reader_open() call.
void vpx_video_reader_close(VpxVideoReader *reader);

// Reads frame from the file and stores it in internal buffer.
int vpx_video_reader_read_frame(VpxVideoReader *reader);

// Returns the pointer to memory buffer with frame data read by last call to
// vpx_video_reader_read_frame().
const uint8_t *vpx_video_reader_get_frame(VpxVideoReader *reader, size_t *size);

// Fills VpxVideoInfo with information from opened video file.
const VpxVideoInfo *vpx_video_reader_get_info(VpxVideoReader *reader);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VIDEO_READER_H_
