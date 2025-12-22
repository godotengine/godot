/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2023 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_AUDIO_AAC_H
#define SPA_AUDIO_AAC_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/param/audio/raw.h>

enum spa_audio_aac_stream_format {
	SPA_AUDIO_AAC_STREAM_FORMAT_UNKNOWN,
	/* Raw AAC frames */
	SPA_AUDIO_AAC_STREAM_FORMAT_RAW,
	/* ISO/IEC 13818-7 MPEG-2 Audio Data Transport Stream (ADTS) */
	SPA_AUDIO_AAC_STREAM_FORMAT_MP2ADTS,
	/* ISO/IEC 14496-3 MPEG-4 Audio Data Transport Stream (ADTS) */
	SPA_AUDIO_AAC_STREAM_FORMAT_MP4ADTS,
	/* ISO/IEC 14496-3 Low Overhead Audio Stream (LOAS) */
	SPA_AUDIO_AAC_STREAM_FORMAT_MP4LOAS,
	/* ISO/IEC 14496-3 Low Overhead Audio Transport Multiplex (LATM) */
	SPA_AUDIO_AAC_STREAM_FORMAT_MP4LATM,
	/* ISO/IEC 14496-3 Audio Data Interchange Format (ADIF) */
	SPA_AUDIO_AAC_STREAM_FORMAT_ADIF,
	/* ISO/IEC 14496-12 MPEG-4 file format */
	SPA_AUDIO_AAC_STREAM_FORMAT_MP4FF,

	SPA_AUDIO_AAC_STREAM_FORMAT_CUSTOM = 0x10000,
};

struct spa_audio_info_aac {
	uint32_t rate;					/*< sample rate */
	uint32_t channels;				/*< number of channels */
	uint32_t bitrate;				/*< stream bitrate */
	enum spa_audio_aac_stream_format stream_format;	/*< AAC audio stream format */
};

#define SPA_AUDIO_INFO_AAC_INIT(...)		((struct spa_audio_info_aac) { __VA_ARGS__ })

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_AUDIO_AAC_H */
