/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_PARAM_ROUTE_TYPES_H
#define SPA_PARAM_ROUTE_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_param
 * \{
 */

#include <spa/utils/enum-types.h>
#include <spa/param/param-types.h>

#include <spa/param/route.h>

#define SPA_TYPE_INFO_PARAM_Route		SPA_TYPE_INFO_PARAM_BASE "Route"
#define SPA_TYPE_INFO_PARAM_ROUTE_BASE		SPA_TYPE_INFO_PARAM_Route ":"

static const struct spa_type_info spa_type_param_route[] = {
	{ SPA_PARAM_ROUTE_START, SPA_TYPE_Id, SPA_TYPE_INFO_PARAM_ROUTE_BASE, spa_type_param, },
	{ SPA_PARAM_ROUTE_index, SPA_TYPE_Int, SPA_TYPE_INFO_PARAM_ROUTE_BASE "index", NULL, },
	{ SPA_PARAM_ROUTE_direction, SPA_TYPE_Id, SPA_TYPE_INFO_PARAM_ROUTE_BASE "direction", spa_type_direction, },
	{ SPA_PARAM_ROUTE_device, SPA_TYPE_Int, SPA_TYPE_INFO_PARAM_ROUTE_BASE "device", NULL, },
	{ SPA_PARAM_ROUTE_name, SPA_TYPE_String, SPA_TYPE_INFO_PARAM_ROUTE_BASE "name", NULL, },
	{ SPA_PARAM_ROUTE_description, SPA_TYPE_String, SPA_TYPE_INFO_PARAM_ROUTE_BASE "description", NULL, },
	{ SPA_PARAM_ROUTE_priority, SPA_TYPE_Int, SPA_TYPE_INFO_PARAM_ROUTE_BASE "priority", NULL, },
	{ SPA_PARAM_ROUTE_available, SPA_TYPE_Id, SPA_TYPE_INFO_PARAM_ROUTE_BASE "available", spa_type_param_availability, },
	{ SPA_PARAM_ROUTE_info, SPA_TYPE_Struct, SPA_TYPE_INFO_PARAM_ROUTE_BASE "info", NULL, },
	{ SPA_PARAM_ROUTE_profiles, SPA_TYPE_Int, SPA_TYPE_INFO_PARAM_ROUTE_BASE "profiles", NULL, },
	{ SPA_PARAM_ROUTE_props, SPA_TYPE_OBJECT_Props, SPA_TYPE_INFO_PARAM_ROUTE_BASE "props", NULL, },
	{ SPA_PARAM_ROUTE_devices, SPA_TYPE_Int, SPA_TYPE_INFO_PARAM_ROUTE_BASE "devices", NULL, },
	{ SPA_PARAM_ROUTE_profile, SPA_TYPE_Int, SPA_TYPE_INFO_PARAM_ROUTE_BASE "profile", NULL, },
	{ SPA_PARAM_ROUTE_save, SPA_TYPE_Bool, SPA_TYPE_INFO_PARAM_ROUTE_BASE "save", NULL, },
	{ 0, 0, NULL, NULL },
};

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_PARAM_ROUTE_TYPES_H */
