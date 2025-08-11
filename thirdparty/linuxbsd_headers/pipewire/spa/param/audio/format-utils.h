/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_PARAM_AUDIO_FORMAT_UTILS_H
#define SPA_PARAM_AUDIO_FORMAT_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/pod/parser.h>
#include <spa/pod/builder.h>
#include <spa/param/audio/format.h>
#include <spa/param/format-utils.h>

#include <spa/param/audio/raw-utils.h>
#include <spa/param/audio/dsp-utils.h>
#include <spa/param/audio/iec958-utils.h>
#include <spa/param/audio/dsd-utils.h>
#include <spa/param/audio/mp3-utils.h>
#include <spa/param/audio/aac-utils.h>
#include <spa/param/audio/vorbis-utils.h>
#include <spa/param/audio/wma-utils.h>
#include <spa/param/audio/ra-utils.h>
#include <spa/param/audio/amr-utils.h>
#include <spa/param/audio/alac-utils.h>
#include <spa/param/audio/flac-utils.h>
#include <spa/param/audio/ape-utils.h>


/**
 * \addtogroup spa_param
 * \{
 */

static inline int
spa_format_audio_parse(const struct spa_pod *format, struct spa_audio_info *info)
{
	int res;

	if ((res = spa_format_parse(format, &info->media_type, &info->media_subtype)) < 0)
		return res;

	if (info->media_type != SPA_MEDIA_TYPE_audio)
		return -EINVAL;

	switch (info->media_subtype) {
	case SPA_MEDIA_SUBTYPE_raw:
		return spa_format_audio_raw_parse(format, &info->info.raw);
	case SPA_MEDIA_SUBTYPE_dsp:
		return spa_format_audio_dsp_parse(format, &info->info.dsp);
	case SPA_MEDIA_SUBTYPE_iec958:
		return spa_format_audio_iec958_parse(format, &info->info.iec958);
	case SPA_MEDIA_SUBTYPE_dsd:
		return spa_format_audio_dsd_parse(format, &info->info.dsd);
	case SPA_MEDIA_SUBTYPE_mp3:
		return spa_format_audio_mp3_parse(format, &info->info.mp3);
	case SPA_MEDIA_SUBTYPE_aac:
		return spa_format_audio_aac_parse(format, &info->info.aac);
	case SPA_MEDIA_SUBTYPE_vorbis:
		return spa_format_audio_vorbis_parse(format, &info->info.vorbis);
	case SPA_MEDIA_SUBTYPE_wma:
		return spa_format_audio_wma_parse(format, &info->info.wma);
	case SPA_MEDIA_SUBTYPE_ra:
		return spa_format_audio_ra_parse(format, &info->info.ra);
	case SPA_MEDIA_SUBTYPE_amr:
		return spa_format_audio_amr_parse(format, &info->info.amr);
	case SPA_MEDIA_SUBTYPE_alac:
		return spa_format_audio_alac_parse(format, &info->info.alac);
	case SPA_MEDIA_SUBTYPE_flac:
		return spa_format_audio_flac_parse(format, &info->info.flac);
	case SPA_MEDIA_SUBTYPE_ape:
		return spa_format_audio_ape_parse(format, &info->info.ape);
	}
	return -ENOTSUP;
}

static inline struct spa_pod *
spa_format_audio_build(struct spa_pod_builder *builder, uint32_t id, struct spa_audio_info *info)
{
	switch (info->media_subtype) {
	case SPA_MEDIA_SUBTYPE_raw:
		return spa_format_audio_raw_build(builder, id, &info->info.raw);
	case SPA_MEDIA_SUBTYPE_dsp:
		return spa_format_audio_dsp_build(builder, id, &info->info.dsp);
	case SPA_MEDIA_SUBTYPE_iec958:
		return spa_format_audio_iec958_build(builder, id, &info->info.iec958);
	case SPA_MEDIA_SUBTYPE_dsd:
		return spa_format_audio_dsd_build(builder, id, &info->info.dsd);
	case SPA_MEDIA_SUBTYPE_mp3:
		return spa_format_audio_mp3_build(builder, id, &info->info.mp3);
	case SPA_MEDIA_SUBTYPE_aac:
		return spa_format_audio_aac_build(builder, id, &info->info.aac);
	case SPA_MEDIA_SUBTYPE_vorbis:
		return spa_format_audio_vorbis_build(builder, id, &info->info.vorbis);
	case SPA_MEDIA_SUBTYPE_wma:
		return spa_format_audio_wma_build(builder, id, &info->info.wma);
	case SPA_MEDIA_SUBTYPE_ra:
		return spa_format_audio_ra_build(builder, id, &info->info.ra);
	case SPA_MEDIA_SUBTYPE_amr:
		return spa_format_audio_amr_build(builder, id, &info->info.amr);
	case SPA_MEDIA_SUBTYPE_alac:
		return spa_format_audio_alac_build(builder, id, &info->info.alac);
	case SPA_MEDIA_SUBTYPE_flac:
		return spa_format_audio_flac_build(builder, id, &info->info.flac);
	case SPA_MEDIA_SUBTYPE_ape:
		return spa_format_audio_ape_build(builder, id, &info->info.ape);
	}
	errno = ENOTSUP;
	return NULL;
}
/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_PARAM_AUDIO_FORMAT_UTILS_H */
