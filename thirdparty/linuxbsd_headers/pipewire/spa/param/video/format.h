/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_PARAM_VIDEO_FORMAT_H
#define SPA_PARAM_VIDEO_FORMAT_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_param
 * \{
 */

#include <spa/param/format.h>
#include <spa/param/video/raw.h>
#include <spa/param/video/dsp.h>
#include <spa/param/video/encoded.h>

struct spa_video_info {
	uint32_t media_type;
	uint32_t media_subtype;
	union {
		struct spa_video_info_raw raw;
		struct spa_video_info_dsp dsp;
		struct spa_video_info_h264 h264;
		struct spa_video_info_mjpg mjpg;
	} info;
};

/**
 * \}
 */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SPA_PARAM_VIDEO_FORMAT_H */
