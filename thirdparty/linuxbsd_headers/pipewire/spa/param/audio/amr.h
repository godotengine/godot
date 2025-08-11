/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2023 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_AUDIO_AMR_H
#define SPA_AUDIO_AMR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/param/audio/raw.h>

enum spa_audio_amr_band_mode {
	SPA_AUDIO_AMR_BAND_MODE_UNKNOWN,
	SPA_AUDIO_AMR_BAND_MODE_NB,
	SPA_AUDIO_AMR_BAND_MODE_WB,
};

struct spa_audio_info_amr {
	uint32_t rate;				/*< sample rate */
	uint32_t channels;			/*< number of channels */
	enum spa_audio_amr_band_mode band_mode;
};

#define SPA_AUDIO_INFO_AMR_INIT(...)		((struct spa_audio_info_amr) { __VA_ARGS__ })

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_AUDIO_AMR_H */
