/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_PARAM_PROFILE_TYPES_H
#define SPA_PARAM_PROFILE_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_param
 * \{
 */

#include <spa/param/param-types.h>

#include <spa/param/profile.h>

#define SPA_TYPE_INFO_PARAM_Profile		SPA_TYPE_INFO_PARAM_BASE "Profile"
#define SPA_TYPE_INFO_PARAM_PROFILE_BASE	SPA_TYPE_INFO_PARAM_Profile ":"

static const struct spa_type_info spa_type_param_profile[] = {
	{ SPA_PARAM_PROFILE_START, SPA_TYPE_Id, SPA_TYPE_INFO_PARAM_PROFILE_BASE, spa_type_param, },
	{ SPA_PARAM_PROFILE_index, SPA_TYPE_Int, SPA_TYPE_INFO_PARAM_PROFILE_BASE "index", NULL },
	{ SPA_PARAM_PROFILE_name, SPA_TYPE_String, SPA_TYPE_INFO_PARAM_PROFILE_BASE "name", NULL },
	{ SPA_PARAM_PROFILE_description, SPA_TYPE_String, SPA_TYPE_INFO_PARAM_PROFILE_BASE "description", NULL },
	{ SPA_PARAM_PROFILE_priority, SPA_TYPE_Int, SPA_TYPE_INFO_PARAM_PROFILE_BASE "priority", NULL },
	{ SPA_PARAM_PROFILE_available, SPA_TYPE_Id, SPA_TYPE_INFO_PARAM_PROFILE_BASE "available", spa_type_param_availability, },
	{ SPA_PARAM_PROFILE_info, SPA_TYPE_Struct, SPA_TYPE_INFO_PARAM_PROFILE_BASE "info", NULL, },
	{ SPA_PARAM_PROFILE_classes, SPA_TYPE_Struct, SPA_TYPE_INFO_PARAM_PROFILE_BASE "classes", NULL, },
	{ SPA_PARAM_PROFILE_save, SPA_TYPE_Bool, SPA_TYPE_INFO_PARAM_PROFILE_BASE "save", NULL, },
	{ 0, 0, NULL, NULL },
};

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_PARAM_PROFILE_TYPES_H */
