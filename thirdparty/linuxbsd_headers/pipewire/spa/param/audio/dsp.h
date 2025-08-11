/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_AUDIO_DSP_H
#define SPA_AUDIO_DSP_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/param/audio/raw.h>

struct spa_audio_info_dsp {
	enum spa_audio_format format;		/*< format, one of the DSP formats in enum spa_audio_format */
};

#define SPA_AUDIO_INFO_DSP_INIT(...)		((struct spa_audio_info_dsp) { __VA_ARGS__ })

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_AUDIO_DSP_H */
