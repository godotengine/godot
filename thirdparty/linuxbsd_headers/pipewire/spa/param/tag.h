/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2023 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_PARAM_TAG_H
#define SPA_PARAM_TAG_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_param
 * \{
 */

#include <spa/param/param.h>

/** properties for SPA_TYPE_OBJECT_ParamTag */
enum spa_param_tag {
	SPA_PARAM_TAG_START,
	SPA_PARAM_TAG_direction,		/**< direction, input/output (Id enum spa_direction) */
	SPA_PARAM_TAG_info,			/**< Struct(
						  *      Int: n_items
						  *      (String: key
						  *       String: value)*
						  *  ) */
};

/** helper structure for managing tag objects */
struct spa_tag_info {
	enum spa_direction direction;
	const struct spa_pod *info;
};

#define SPA_TAG_INFO(dir,...) ((struct spa_tag_info) { .direction = (dir), ## __VA_ARGS__ })

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_PARAM_TAG_H */
