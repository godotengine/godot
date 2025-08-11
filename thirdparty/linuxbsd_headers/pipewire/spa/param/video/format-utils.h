/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_PARAM_VIDEO_FORMAT_UTILS_H
#define SPA_PARAM_VIDEO_FORMAT_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/param/format-utils.h>
#include <spa/param/video/format.h>
#include <spa/param/video/raw-utils.h>
#include <spa/param/video/dsp-utils.h>
#include <spa/param/video/h264-utils.h>
#include <spa/param/video/mjpg-utils.h>

static inline int
spa_format_video_parse(const struct spa_pod *format, struct spa_video_info *info)
{
	int res;

	if ((res = spa_format_parse(format, &info->media_type, &info->media_subtype)) < 0)
		return res;

	if (info->media_type != SPA_MEDIA_TYPE_video)
		return -EINVAL;

	switch (info->media_subtype) {
	case SPA_MEDIA_SUBTYPE_raw:
		return spa_format_video_raw_parse(format, &info->info.raw);
	case SPA_MEDIA_SUBTYPE_dsp:
		return spa_format_video_dsp_parse(format, &info->info.dsp);
	case SPA_MEDIA_SUBTYPE_h264:
		return spa_format_video_h264_parse(format, &info->info.h264);
	case SPA_MEDIA_SUBTYPE_mjpg:
		return spa_format_video_mjpg_parse(format, &info->info.mjpg);
	}
	return -ENOTSUP;
}

static inline struct spa_pod *
spa_format_video_build(struct spa_pod_builder *builder, uint32_t id, struct spa_video_info *info)
{
	switch (info->media_subtype) {
	case SPA_MEDIA_SUBTYPE_raw:
		return spa_format_video_raw_build(builder, id, &info->info.raw);
	case SPA_MEDIA_SUBTYPE_dsp:
		return spa_format_video_dsp_build(builder, id, &info->info.dsp);
	case SPA_MEDIA_SUBTYPE_h264:
		return spa_format_video_h264_build(builder, id, &info->info.h264);
	case SPA_MEDIA_SUBTYPE_mjpg:
		return spa_format_video_mjpg_build(builder, id, &info->info.mjpg);
	}
	errno = ENOTSUP;
	return NULL;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SPA_PARAM_VIDEO_FORMAT_UTILS_H */
