/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_AUDIO_AMR_TYPES_H
#define SPA_AUDIO_AMR_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/utils/type.h>
#include <spa/param/audio/amr.h>

#define SPA_TYPE_INFO_AudioAMRBandMode		SPA_TYPE_INFO_ENUM_BASE "AudioAMRBandMode"
#define SPA_TYPE_INFO_AUDIO_AMR_BAND_MODE_BASE	SPA_TYPE_INFO_AudioAMRBandMode ":"

static const struct spa_type_info spa_type_audio_amr_band_mode[] = {
	{ SPA_AUDIO_AMR_BAND_MODE_UNKNOWN, SPA_TYPE_Int, SPA_TYPE_INFO_AUDIO_AMR_BAND_MODE_BASE "UNKNOWN", NULL },
	{ SPA_AUDIO_AMR_BAND_MODE_NB, SPA_TYPE_Int, SPA_TYPE_INFO_AUDIO_AMR_BAND_MODE_BASE "NB", NULL },
	{ SPA_AUDIO_AMR_BAND_MODE_WB, SPA_TYPE_Int, SPA_TYPE_INFO_AUDIO_AMR_BAND_MODE_BASE "WB", NULL },
	{ 0, 0, NULL, NULL },
};
/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_AUDIO_AMR_TYPES_H */
