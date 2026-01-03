/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_AUDIO_DSD_UTILS_H
#define SPA_AUDIO_DSD_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_param
 * \{
 */

#include <spa/pod/parser.h>
#include <spa/pod/builder.h>
#include <spa/param/format-utils.h>
#include <spa/param/audio/format.h>

static inline int
spa_format_audio_dsd_parse(const struct spa_pod *format, struct spa_audio_info_dsd *info)
{
	struct spa_pod *position = NULL;
	int res;
	info->flags = 0;
	res = spa_pod_parse_object(format,
			SPA_TYPE_OBJECT_Format, NULL,
			SPA_FORMAT_AUDIO_bitorder,	SPA_POD_OPT_Id(&info->bitorder),
			SPA_FORMAT_AUDIO_interleave,	SPA_POD_OPT_Int(&info->interleave),
			SPA_FORMAT_AUDIO_rate,		SPA_POD_OPT_Int(&info->rate),
			SPA_FORMAT_AUDIO_channels,	SPA_POD_OPT_Int(&info->channels),
			SPA_FORMAT_AUDIO_position,	SPA_POD_OPT_Pod(&position));
	if (position == NULL ||
	    !spa_pod_copy_array(position, SPA_TYPE_Id, info->position, SPA_AUDIO_MAX_CHANNELS))
		SPA_FLAG_SET(info->flags, SPA_AUDIO_FLAG_UNPOSITIONED);

	return res;
}

static inline struct spa_pod *
spa_format_audio_dsd_build(struct spa_pod_builder *builder, uint32_t id, struct spa_audio_info_dsd *info)
{
	struct spa_pod_frame f;
	spa_pod_builder_push_object(builder, &f, SPA_TYPE_OBJECT_Format, id);
	spa_pod_builder_add(builder,
			SPA_FORMAT_mediaType,		SPA_POD_Id(SPA_MEDIA_TYPE_audio),
			SPA_FORMAT_mediaSubtype,	SPA_POD_Id(SPA_MEDIA_SUBTYPE_dsd),
			0);
	if (info->bitorder != SPA_PARAM_BITORDER_unknown)
		spa_pod_builder_add(builder,
			SPA_FORMAT_AUDIO_bitorder,	SPA_POD_Id(info->bitorder), 0);
	if (info->interleave != 0)
		spa_pod_builder_add(builder,
			SPA_FORMAT_AUDIO_interleave,	SPA_POD_Int(info->interleave), 0);
	if (info->rate != 0)
		spa_pod_builder_add(builder,
			SPA_FORMAT_AUDIO_rate,		SPA_POD_Int(info->rate), 0);
	if (info->channels != 0) {
		spa_pod_builder_add(builder,
			SPA_FORMAT_AUDIO_channels,	SPA_POD_Int(info->channels), 0);
		if (!SPA_FLAG_IS_SET(info->flags, SPA_AUDIO_FLAG_UNPOSITIONED)) {
			spa_pod_builder_add(builder, SPA_FORMAT_AUDIO_position,
				SPA_POD_Array(sizeof(uint32_t), SPA_TYPE_Id,
					info->channels, info->position), 0);
		}
	}
	return (struct spa_pod*)spa_pod_builder_pop(builder, &f);
}

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_AUDIO_DSD_UTILS_H */
