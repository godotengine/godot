/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_VIDEO_MJPG_H
#define SPA_VIDEO_MJPG_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_param
 * \{
 */

#include <spa/param/format.h>

struct spa_video_info_mjpg {
	struct spa_rectangle size;
	struct spa_fraction framerate;
	struct spa_fraction max_framerate;
};

/**
 * \}
 */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SPA_VIDEO_MJPG_H */
