/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2023 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_AUDIO_WMA_H
#define SPA_AUDIO_WMA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/param/audio/raw.h>

enum spa_audio_wma_profile {
	SPA_AUDIO_WMA_PROFILE_UNKNOWN,

	SPA_AUDIO_WMA_PROFILE_WMA7,
	SPA_AUDIO_WMA_PROFILE_WMA8,
	SPA_AUDIO_WMA_PROFILE_WMA9,
	SPA_AUDIO_WMA_PROFILE_WMA10,
	SPA_AUDIO_WMA_PROFILE_WMA9_PRO,
	SPA_AUDIO_WMA_PROFILE_WMA9_LOSSLESS,
	SPA_AUDIO_WMA_PROFILE_WMA10_LOSSLESS,

	SPA_AUDIO_WMA_PROFILE_CUSTOM = 0x10000,
};

struct spa_audio_info_wma {
	uint32_t rate;				/*< sample rate */
	uint32_t channels;			/*< number of channels */
	uint32_t bitrate;			/*< stream bitrate */
	uint32_t block_align;			/*< block alignment */
	enum spa_audio_wma_profile profile;	/*< WMA profile */

};

#define SPA_AUDIO_INFO_WMA_INIT(...)		((struct spa_audio_info_wma) { __VA_ARGS__ })

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_AUDIO_WMA_H */
