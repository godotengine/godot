/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_PARAM_AUDIO_FORMAT_H
#define SPA_PARAM_AUDIO_FORMAT_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_param
 * \{
 */

#include <spa/param/format.h>
#include <spa/param/audio/raw.h>
#include <spa/param/audio/dsp.h>
#include <spa/param/audio/iec958.h>
#include <spa/param/audio/dsd.h>
#include <spa/param/audio/mp3.h>
#include <spa/param/audio/aac.h>
#include <spa/param/audio/vorbis.h>
#include <spa/param/audio/wma.h>
#include <spa/param/audio/ra.h>
#include <spa/param/audio/amr.h>
#include <spa/param/audio/alac.h>
#include <spa/param/audio/flac.h>
#include <spa/param/audio/ape.h>
#include <spa/param/audio/opus.h>

struct spa_audio_info {
	uint32_t media_type;
	uint32_t media_subtype;
	union {
		struct spa_audio_info_raw raw;
		struct spa_audio_info_dsp dsp;
		struct spa_audio_info_iec958 iec958;
		struct spa_audio_info_dsd dsd;
		struct spa_audio_info_mp3 mp3;
		struct spa_audio_info_aac aac;
		struct spa_audio_info_vorbis vorbis;
		struct spa_audio_info_wma wma;
		struct spa_audio_info_ra ra;
		struct spa_audio_info_amr amr;
		struct spa_audio_info_alac alac;
		struct spa_audio_info_flac flac;
		struct spa_audio_info_ape ape;
		struct spa_audio_info_ape opus;
	} info;
};

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_PARAM_AUDIO_FORMAT_H */
