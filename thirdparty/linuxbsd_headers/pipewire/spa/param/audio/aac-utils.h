/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2023 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_AUDIO_AAC_UTILS_H
#define SPA_AUDIO_AAC_UTILS_H

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
spa_format_audio_aac_parse(const struct spa_pod *format, struct spa_audio_info_aac *info)
{
	int res;
	res = spa_pod_parse_object(format,
			SPA_TYPE_OBJECT_Format, NULL,
			SPA_FORMAT_AUDIO_rate,		SPA_POD_OPT_Int(&info->rate),
			SPA_FORMAT_AUDIO_channels,	SPA_POD_OPT_Int(&info->channels),
			SPA_FORMAT_AUDIO_bitrate,	SPA_POD_OPT_Int(&info->bitrate),
			SPA_FORMAT_AUDIO_AAC_streamFormat,	SPA_POD_OPT_Id(&info->stream_format));

	return res;
}

static inline struct spa_pod *
spa_format_audio_aac_build(struct spa_pod_builder *builder, uint32_t id,
			   const struct spa_audio_info_aac *info)
{
	struct spa_pod_frame f;
	spa_pod_builder_push_object(builder, &f, SPA_TYPE_OBJECT_Format, id);
	spa_pod_builder_add(builder,
			SPA_FORMAT_mediaType,		SPA_POD_Id(SPA_MEDIA_TYPE_audio),
			SPA_FORMAT_mediaSubtype,	SPA_POD_Id(SPA_MEDIA_SUBTYPE_aac),
			SPA_FORMAT_AUDIO_format,	SPA_POD_Id(SPA_AUDIO_FORMAT_ENCODED),
			0);
	if (info->rate != 0)
		spa_pod_builder_add(builder,
			SPA_FORMAT_AUDIO_rate,		SPA_POD_Int(info->rate), 0);
	if (info->channels != 0)
		spa_pod_builder_add(builder,
			SPA_FORMAT_AUDIO_channels,	SPA_POD_Int(info->channels), 0);
	if (info->bitrate != 0)
		spa_pod_builder_add(builder,
			SPA_FORMAT_AUDIO_bitrate,	SPA_POD_Int(info->bitrate), 0);
	if (info->stream_format != 0)
		spa_pod_builder_add(builder,
			SPA_FORMAT_AUDIO_AAC_streamFormat,	SPA_POD_Id(info->stream_format), 0);
	return (struct spa_pod*)spa_pod_builder_pop(builder, &f);
}

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_AUDIO_AAC_UTILS_H */
