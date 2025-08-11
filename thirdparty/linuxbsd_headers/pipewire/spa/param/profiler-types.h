/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_PARAM_PROFILER_TYPES_H
#define SPA_PARAM_PROFILER_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_param
 * \{
 */

#include <spa/param/param-types.h>
#include <spa/param/profiler.h>

#define SPA_TYPE_INFO_Profiler		SPA_TYPE_INFO_OBJECT_BASE "Profiler"
#define SPA_TYPE_INFO_PROFILER_BASE	SPA_TYPE_INFO_Profiler ":"

static const struct spa_type_info spa_type_profiler[] = {
	{ SPA_PROFILER_START, SPA_TYPE_Id, SPA_TYPE_INFO_PROFILER_BASE, spa_type_param, },
	{ SPA_PROFILER_info, SPA_TYPE_Struct, SPA_TYPE_INFO_PROFILER_BASE "info", NULL, },
	{ SPA_PROFILER_clock, SPA_TYPE_Struct, SPA_TYPE_INFO_PROFILER_BASE "clock", NULL, },
	{ SPA_PROFILER_driverBlock, SPA_TYPE_Struct, SPA_TYPE_INFO_PROFILER_BASE "driverBlock", NULL, },
	{ SPA_PROFILER_followerBlock, SPA_TYPE_Struct, SPA_TYPE_INFO_PROFILER_BASE "followerBlock", NULL, },
	{ 0, 0, NULL, NULL },
};

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_PARAM_PROFILER_TYPES_H */
