/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_AUDIO_MP3_TYPES_H
#define SPA_AUDIO_MP3_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/utils/type.h>
#include <spa/param/audio/mp3.h>

#define SPA_TYPE_INFO_AudioMP3ChannelMode		SPA_TYPE_INFO_ENUM_BASE "AudioMP3ChannelMode"
#define SPA_TYPE_INFO_AUDIO_MP3_CHANNEL_MODE_BASE	SPA_TYPE_INFO_AudioMP3ChannelMode ":"

static const struct spa_type_info spa_type_audio_mp3_channel_mode[] = {
	{ SPA_AUDIO_MP3_CHANNEL_MODE_UNKNOWN, SPA_TYPE_Int, SPA_TYPE_INFO_AUDIO_MP3_CHANNEL_MODE_BASE "UNKNOWN", NULL },
	{ SPA_AUDIO_MP3_CHANNEL_MODE_MONO, SPA_TYPE_Int, SPA_TYPE_INFO_AUDIO_MP3_CHANNEL_MODE_BASE "Mono", NULL },
	{ SPA_AUDIO_MP3_CHANNEL_MODE_STEREO, SPA_TYPE_Int, SPA_TYPE_INFO_AUDIO_MP3_CHANNEL_MODE_BASE "Stereo", NULL },
	{ SPA_AUDIO_MP3_CHANNEL_MODE_JOINTSTEREO, SPA_TYPE_Int, SPA_TYPE_INFO_AUDIO_MP3_CHANNEL_MODE_BASE "Joint-stereo", NULL },
	{ SPA_AUDIO_MP3_CHANNEL_MODE_DUAL, SPA_TYPE_Int, SPA_TYPE_INFO_AUDIO_MP3_CHANNEL_MODE_BASE "Dual", NULL },
	{ 0, 0, NULL, NULL },
};
/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_AUDIO_MP3_TYPES_H */
