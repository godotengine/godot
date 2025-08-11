/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2023 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_AUDIO_MP3_H
#define SPA_AUDIO_MP3_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/param/audio/raw.h>

enum spa_audio_mp3_channel_mode {
	SPA_AUDIO_MP3_CHANNEL_MODE_UNKNOWN,
	SPA_AUDIO_MP3_CHANNEL_MODE_MONO,
	SPA_AUDIO_MP3_CHANNEL_MODE_STEREO,
	SPA_AUDIO_MP3_CHANNEL_MODE_JOINTSTEREO,
	SPA_AUDIO_MP3_CHANNEL_MODE_DUAL,
};

struct spa_audio_info_mp3 {
	uint32_t rate;				/*< sample rate */
	uint32_t channels;			/*< number of channels */
};

#define SPA_AUDIO_INFO_MP3_INIT(...)		((struct spa_audio_info_mp3) { __VA_ARGS__ })

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_AUDIO_MP3_H */
