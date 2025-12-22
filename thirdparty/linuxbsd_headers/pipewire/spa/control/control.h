/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_CONTROL_H
#define SPA_CONTROL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/utils/defs.h>
#include <spa/pod/pod.h>

/** \defgroup spa_control Control
 * Control type declarations
 */

/**
 * \addtogroup spa_control
 * \{
 */

/** Different Control types */
enum spa_control_type {
	SPA_CONTROL_Invalid,
	SPA_CONTROL_Properties,		/**< data contains a SPA_TYPE_OBJECT_Props */
	SPA_CONTROL_Midi,		/**< data contains a spa_pod_bytes with raw midi data */
	SPA_CONTROL_OSC,		/**< data contains a spa_pod_bytes with an OSC packet */

	_SPA_CONTROL_LAST,		/**< not part of ABI */
};

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_CONTROL_H */
