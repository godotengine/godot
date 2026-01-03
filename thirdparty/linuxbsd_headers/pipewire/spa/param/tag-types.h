/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_PARAM_TAG_TYPES_H
#define SPA_PARAM_TAG_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_param
 * \{
 */

#include <spa/utils/enum-types.h>
#include <spa/param/param-types.h>
#include <spa/param/tag.h>

#define SPA_TYPE_INFO_PARAM_Tag		SPA_TYPE_INFO_PARAM_BASE "Tag"
#define SPA_TYPE_INFO_PARAM_TAG_BASE	SPA_TYPE_INFO_PARAM_Tag ":"

static const struct spa_type_info spa_type_param_tag[] = {
	{ SPA_PARAM_TAG_START, SPA_TYPE_Id, SPA_TYPE_INFO_PARAM_TAG_BASE, spa_type_param, },
	{ SPA_PARAM_TAG_direction, SPA_TYPE_Id, SPA_TYPE_INFO_PARAM_TAG_BASE "direction", spa_type_direction, },
	{ SPA_PARAM_TAG_info, SPA_TYPE_Struct, SPA_TYPE_INFO_PARAM_TAG_BASE "info", NULL, },
	{ 0, 0, NULL, NULL },
};

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_PARAM_TAG_TYPES_H */
