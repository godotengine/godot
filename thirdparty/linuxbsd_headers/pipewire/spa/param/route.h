/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_PARAM_ROUTE_H
#define SPA_PARAM_ROUTE_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_param
 * \{
 */

#include <spa/param/param.h>

/** properties for SPA_TYPE_OBJECT_ParamRoute */
enum spa_param_route {
	SPA_PARAM_ROUTE_START,
	SPA_PARAM_ROUTE_index,			/**< index of the routing destination (Int) */
	SPA_PARAM_ROUTE_direction,		/**< direction, input/output (Id enum spa_direction) */
	SPA_PARAM_ROUTE_device,			/**< device id (Int) */
	SPA_PARAM_ROUTE_name,			/**< name of the routing destination (String) */
	SPA_PARAM_ROUTE_description,		/**< description of the destination (String) */
	SPA_PARAM_ROUTE_priority,		/**< priority of the destination (Int) */
	SPA_PARAM_ROUTE_available,		/**< availability of the destination
						  *  (Id enum spa_param_availability) */
	SPA_PARAM_ROUTE_info,			/**< info (Struct(
						  *		  Int : n_items,
						  *		  (String : key,
						  *		   String : value)*)) */
	SPA_PARAM_ROUTE_profiles,		/**< associated profile indexes (Array of Int) */
	SPA_PARAM_ROUTE_props,			/**< properties SPA_TYPE_OBJECT_Props */
	SPA_PARAM_ROUTE_devices,		/**< associated device indexes (Array of Int) */
	SPA_PARAM_ROUTE_profile,		/**< profile id (Int) */
	SPA_PARAM_ROUTE_save,			/**< If route should be saved (Bool) */
};

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_PARAM_ROUTE_H */
