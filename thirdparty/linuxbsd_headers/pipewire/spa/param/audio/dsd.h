/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2021 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_AUDIO_DSD_H
#define SPA_AUDIO_DSD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/param/param.h>
#include <spa/param/audio/raw.h>

/**
 * \addtogroup spa_param
 * \{
 */

/** Extra DSD audio flags */
#define SPA_AUDIO_DSD_FLAG_NONE		(0)		/*< no valid flag */

/* DSD bits are transferred in a buffer grouped in bytes with the bitorder
 * defined by \a bitorder.
 *
 * Channels are placed in separate planes (interleave = 0) or interleaved
 * using the interleave value. A negative interleave value means that the
 * bytes need to be reversed in the group.
 *
 *  Planar (interleave = 0):
 *    plane1: l1 l2 l3 l4 l5 ...
 *    plane2: r1 r2 r3 r4 r5 ...
 *
 *  Interleaved 4:
 *    plane1: l1 l2 l3 l4 r1 r2 r3 r4 l5 l6 l7 l8 r5 r6 r7 r8 l9 ...
 *
 *  Interleaved 2:
 *    plane1: l1 l2 r1 r2 l3 l4 r3 r4  ...
 */
struct spa_audio_info_dsd {
	enum spa_param_bitorder bitorder;		/*< the order of the bits */
	uint32_t flags;					/*< extra flags */
	int32_t interleave;				/*< interleave bytes */
	uint32_t rate;					/*< sample rate (in bytes per second) */
	uint32_t channels;				/*< channels */
	uint32_t position[SPA_AUDIO_MAX_CHANNELS];	/*< channel position from enum spa_audio_channel */
};

#define SPA_AUDIO_INFO_DSD_INIT(...)		((struct spa_audio_info_dsd) { __VA_ARGS__ })

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_AUDIO_DSD_H */
