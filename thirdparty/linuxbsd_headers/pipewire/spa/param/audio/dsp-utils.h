/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_AUDIO_DSP_UTILS_H
#define SPA_AUDIO_DSP_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_param
 * \{
 */

#include <spa/pod/parser.h>
#include <spa/pod/builder.h>
#include <spa/param/audio/format.h>
#include <spa/param/format-utils.h>

static inline int
spa_format_audio_dsp_parse(const struct spa_pod *format, struct spa_audio_info_dsp *info)
{
	int res;
	res = spa_pod_parse_object(format,
			SPA_TYPE_OBJECT_Format, NULL,
			SPA_FORMAT_AUDIO_format,	SPA_POD_OPT_Id(&info->format));
	return res;
}

static inline struct spa_pod *
spa_format_audio_dsp_build(struct spa_pod_builder *builder, uint32_t id, struct spa_audio_info_dsp *info)
{
	struct spa_pod_frame f;
	spa_pod_builder_push_object(builder, &f, SPA_TYPE_OBJECT_Format, id);
	spa_pod_builder_add(builder,
			SPA_FORMAT_mediaType,		SPA_POD_Id(SPA_MEDIA_TYPE_audio),
			SPA_FORMAT_mediaSubtype,	SPA_POD_Id(SPA_MEDIA_SUBTYPE_dsp),
			0);
	if (info->format != SPA_AUDIO_FORMAT_UNKNOWN)
		spa_pod_builder_add(builder,
			SPA_FORMAT_AUDIO_format,	SPA_POD_Id(info->format), 0);
	return (struct spa_pod*)spa_pod_builder_pop(builder, &f);
}

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_AUDIO_DSP_UTILS_H */
