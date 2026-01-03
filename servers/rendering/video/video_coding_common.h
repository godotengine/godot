/**************************************************************************/
/*  video_coding_common.h                                                 */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "video_coding_av1.h"
#include "video_coding_av1_decode.h"
#include "video_coding_h264.h"
#include "video_coding_h264_decode.h"

#include <cstdint>

enum VideoCodingOperation {
	VIDEO_OPERATION_DECODE_H264 = (1 << 0),
	VIDEO_OPERATION_DECODE_AV1 = (1 << 2),
};

enum VideoCodingChromaSubsampling {
	VIDEO_CODING_CHROMA_SUBSAMPLING_MONOCHROME = (1 << 0),
	VIDEO_CODING_CHROMA_SUBSAMPLING_420 = (1 << 1),
	VIDEO_CODING_CHROMA_SUBSAMPLING_422 = (1 << 2),
	VIDEO_CODING_CHROMA_SUBSAMPLING_444 = (1 << 3),
};

struct VideoProfile {
	VideoCodingOperation operation;
	VideoCodingChromaSubsampling chroma_subsampling = VIDEO_CODING_CHROMA_SUBSAMPLING_420;
	uint32_t luma_bit_depth = 8;
	uint32_t chroma_bit_depth = 8;

	VideoCodingH264ProfileIdc h264_profile_idc;
	VideoCodingH264PictureLayout h264_picture_layout;
};
