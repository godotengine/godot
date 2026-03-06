/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_AUDIO_IEC958_UTILS_H
#define SPA_AUDIO_IEC958_UTILS_H

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
spa_format_audio_iec958_parse(const struct spa_pod *format, struct spa_audio_info_iec958 *info)
{
	int res;
	res = spa_pod_parse_object(format,
			SPA_TYPE_OBJECT_Format, NULL,
			SPA_FORMAT_AUDIO_iec958Codec,	SPA_POD_OPT_Id(&info->codec),
			SPA_FORMAT_AUDIO_rate,		SPA_POD_OPT_Int(&info->rate));
	return res;
}

static inline struct spa_pod *
spa_format_audio_iec958_build(struct spa_pod_builder *builder, uint32_t id, struct spa_audio_info_iec958 *info)
{
	struct spa_pod_frame f;
	spa_pod_builder_push_object(builder, &f, SPA_TYPE_OBJECT_Format, id);
	spa_pod_builder_add(builder,
			SPA_FORMAT_mediaType,		SPA_POD_Id(SPA_MEDIA_TYPE_audio),
			SPA_FORMAT_mediaSubtype,	SPA_POD_Id(SPA_MEDIA_SUBTYPE_iec958),
			0);
	if (info->codec != SPA_AUDIO_IEC958_CODEC_UNKNOWN)
		spa_pod_builder_add(builder,
			SPA_FORMAT_AUDIO_iec958Codec,	SPA_POD_Id(info->codec), 0);
	if (info->rate != 0)
		spa_pod_builder_add(builder,
			SPA_FORMAT_AUDIO_rate,		SPA_POD_Int(info->rate), 0);
	return (struct spa_pod*)spa_pod_builder_pop(builder, &f);
}

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_AUDIO_IEC958_UTILS_H */
