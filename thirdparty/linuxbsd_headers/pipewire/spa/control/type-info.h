/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_CONTROL_TYPES_H
#define SPA_CONTROL_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_control
 * \{
 */

#include <spa/utils/defs.h>
#include <spa/utils/type-info.h>
#include <spa/control/control.h>

/* base for parameter object enumerations */
#define SPA_TYPE_INFO_Control		SPA_TYPE_INFO_ENUM_BASE "Control"
#define SPA_TYPE_INFO_CONTROL_BASE		SPA_TYPE_INFO_Control ":"

static const struct spa_type_info spa_type_control[] = {
	{ SPA_CONTROL_Invalid, SPA_TYPE_Int, SPA_TYPE_INFO_CONTROL_BASE "Invalid", NULL },
	{ SPA_CONTROL_Properties, SPA_TYPE_Int, SPA_TYPE_INFO_CONTROL_BASE "Properties", NULL },
	{ SPA_CONTROL_Midi, SPA_TYPE_Int, SPA_TYPE_INFO_CONTROL_BASE "Midi", NULL },
	{ SPA_CONTROL_OSC, SPA_TYPE_Int, SPA_TYPE_INFO_CONTROL_BASE "OSC", NULL },
	{ 0, 0, NULL, NULL },
};

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_CONTROL_TYPES_H */
