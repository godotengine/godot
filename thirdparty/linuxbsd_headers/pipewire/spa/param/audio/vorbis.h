/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2023 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_AUDIO_VORBIS_H
#define SPA_AUDIO_VORBIS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/param/audio/raw.h>

struct spa_audio_info_vorbis {
	uint32_t rate;				/*< sample rate */
	uint32_t channels;			/*< number of channels */
};

#define SPA_AUDIO_INFO_VORBIS_INIT(...)		((struct spa_audio_info_vorbis) { __VA_ARGS__ })

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_AUDIO_VORBIS_H */
