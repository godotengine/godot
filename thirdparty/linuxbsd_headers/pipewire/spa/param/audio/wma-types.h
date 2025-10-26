/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_AUDIO_WMA_TYPES_H
#define SPA_AUDIO_WMA_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/utils/type.h>
#include <spa/param/audio/wma.h>

/**
 * \addtogroup spa_param
 * \{
 */

#define SPA_TYPE_INFO_AudioWMAProfile		SPA_TYPE_INFO_ENUM_BASE "AudioWMAProfile"
#define SPA_TYPE_INFO_AUDIO_WMA_PROFILE_BASE	SPA_TYPE_INFO_AudioWMAProfile ":"

static const struct spa_type_info spa_type_audio_wma_profile[] = {
	{ SPA_AUDIO_WMA_PROFILE_UNKNOWN, SPA_TYPE_Int, SPA_TYPE_INFO_AUDIO_WMA_PROFILE_BASE "UNKNOWN", NULL },
	{ SPA_AUDIO_WMA_PROFILE_WMA7, SPA_TYPE_Int, SPA_TYPE_INFO_AUDIO_WMA_PROFILE_BASE "WMA7", NULL },
	{ SPA_AUDIO_WMA_PROFILE_WMA8, SPA_TYPE_Int, SPA_TYPE_INFO_AUDIO_WMA_PROFILE_BASE "WMA8", NULL },
	{ SPA_AUDIO_WMA_PROFILE_WMA9, SPA_TYPE_Int, SPA_TYPE_INFO_AUDIO_WMA_PROFILE_BASE "WMA9", NULL },
	{ SPA_AUDIO_WMA_PROFILE_WMA10, SPA_TYPE_Int, SPA_TYPE_INFO_AUDIO_WMA_PROFILE_BASE "WMA10", NULL },
	{ SPA_AUDIO_WMA_PROFILE_WMA9_PRO, SPA_TYPE_Int, SPA_TYPE_INFO_AUDIO_WMA_PROFILE_BASE "WMA9-Pro", NULL },
	{ SPA_AUDIO_WMA_PROFILE_WMA9_LOSSLESS, SPA_TYPE_Int, SPA_TYPE_INFO_AUDIO_WMA_PROFILE_BASE "WMA9-Lossless", NULL },
	{ SPA_AUDIO_WMA_PROFILE_WMA10_LOSSLESS, SPA_TYPE_Int, SPA_TYPE_INFO_AUDIO_WMA_PROFILE_BASE "WMA10-Lossless", NULL },
	{ 0, 0, NULL, NULL },
};
/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_AUDIO_WMA_TYPES_H */
