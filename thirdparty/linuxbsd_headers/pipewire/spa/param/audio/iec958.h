/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2021 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_AUDIO_IEC958_H
#define SPA_AUDIO_IEC958_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_param
 * \{
 */
enum spa_audio_iec958_codec {
	SPA_AUDIO_IEC958_CODEC_UNKNOWN,

	SPA_AUDIO_IEC958_CODEC_PCM,
	SPA_AUDIO_IEC958_CODEC_DTS,
	SPA_AUDIO_IEC958_CODEC_AC3,
	SPA_AUDIO_IEC958_CODEC_MPEG,		/**< MPEG-1 or MPEG-2 (Part 3, not AAC) */
	SPA_AUDIO_IEC958_CODEC_MPEG2_AAC,	/**< MPEG-2 AAC */

	SPA_AUDIO_IEC958_CODEC_EAC3,

	SPA_AUDIO_IEC958_CODEC_TRUEHD,		/**< Dolby TrueHD */
	SPA_AUDIO_IEC958_CODEC_DTSHD,		/**< DTS-HD Master Audio */
};

struct spa_audio_info_iec958 {
	enum spa_audio_iec958_codec codec;	/*< format, one of the DSP formats in enum spa_audio_format_dsp */
	uint32_t flags;				/*< extra flags */
	uint32_t rate;				/*< sample rate */
};

#define SPA_AUDIO_INFO_IEC958_INIT(...)		((struct spa_audio_info_iec958) { __VA_ARGS__ })

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_AUDIO_IEC958_H */
