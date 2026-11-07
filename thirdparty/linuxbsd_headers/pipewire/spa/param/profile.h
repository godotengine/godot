/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_PARAM_PROFILE_H
#define SPA_PARAM_PROFILE_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_param
 * \{
 */

#include <spa/param/param.h>

/** properties for SPA_TYPE_OBJECT_ParamProfile */
enum spa_param_profile {
	SPA_PARAM_PROFILE_START,
	SPA_PARAM_PROFILE_index,	/**< profile index (Int) */
	SPA_PARAM_PROFILE_name,		/**< profile name (String) */
	SPA_PARAM_PROFILE_description,	/**< profile description (String) */
	SPA_PARAM_PROFILE_priority,	/**< profile priority (Int) */
	SPA_PARAM_PROFILE_available,	/**< availability of the profile
					  *  (Id enum spa_param_availability) */
	SPA_PARAM_PROFILE_info,		/**< info (Struct(
					  *		  Int : n_items,
					  *		  (String : key,
					  *		   String : value)*)) */
	SPA_PARAM_PROFILE_classes,	/**< node classes provided by this profile
					  *  (Struct(
					  *	   Int : number of items following
					  *        Struct(
					  *           String : class name (eg. "Audio/Source"),
					  *           Int : number of nodes
					  *           String : property (eg. "card.profile.devices"),
					  *           Array of Int: device indexes
					  *         )*)) */
	SPA_PARAM_PROFILE_save,		/**< If profile should be saved (Bool) */
};

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_PARAM_PROFILE_H */
