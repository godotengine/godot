/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2023 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_VIDEO_H264_H
#define SPA_VIDEO_H264_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_param
 * \{
 */

#include <spa/param/format.h>

enum spa_h264_stream_format {
	SPA_H264_STREAM_FORMAT_UNKNOWN = 0,
	SPA_H264_STREAM_FORMAT_AVC,
	SPA_H264_STREAM_FORMAT_AVC3,
	SPA_H264_STREAM_FORMAT_BYTESTREAM
};

enum spa_h264_alignment {
	SPA_H264_ALIGNMENT_UNKNOWN = 0,
	SPA_H264_ALIGNMENT_AU,
	SPA_H264_ALIGNMENT_NAL
};

struct spa_video_info_h264 {
	struct spa_rectangle size;
	struct spa_fraction framerate;
	struct spa_fraction max_framerate;
	enum spa_h264_stream_format stream_format;
	enum spa_h264_alignment alignment;
};

/**
 * \}
 */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SPA_VIDEO_H264_H */
