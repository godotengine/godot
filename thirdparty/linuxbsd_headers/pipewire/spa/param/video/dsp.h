/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_VIDEO_DSP_H
#define SPA_VIDEO_DSP_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_param
 * \{
 */

#include <spa/param/video/raw.h>

struct spa_video_info_dsp {
	enum spa_video_format format;
	uint32_t flags;
	uint64_t modifier;
};

#define SPA_VIDEO_INFO_DSP_INIT(...)	((struct spa_video_info_dsp) { __VA_ARGS__ })

/**
 * \}
 */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SPA_VIDEO_DSP_H */
