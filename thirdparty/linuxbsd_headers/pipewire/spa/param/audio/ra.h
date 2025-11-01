/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2023 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_AUDIO_RA_H
#define SPA_AUDIO_RA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/param/audio/raw.h>

/**
 * \addtogroup spa_param
 * \{
 */

struct spa_audio_info_ra {
	uint32_t rate;				/*< sample rate */
	uint32_t channels;			/*< number of channels */
};

#define SPA_AUDIO_INFO_RA_INIT(...)		((struct spa_audio_info_ra) { __VA_ARGS__ })

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_AUDIO_RA_H */
