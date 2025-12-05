/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_DEBUG_DICT_H
#define SPA_DEBUG_DICT_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_debug
 * \{
 */

#include <spa/debug/context.h>
#include <spa/utils/dict.h>

static inline int spa_debugc_dict(struct spa_debug_context *ctx, int indent, const struct spa_dict *dict)
{
	const struct spa_dict_item *item;
	spa_debugc(ctx, "%*sflags:%08x n_items:%d", indent, "", dict->flags, dict->n_items);
	spa_dict_for_each(item, dict) {
		spa_debugc(ctx, "%*s  %s = \"%s\"", indent, "", item->key, item->value);
	}
	return 0;
}

static inline int spa_debug_dict(int indent, const struct spa_dict *dict)
{
	return spa_debugc_dict(NULL, indent, dict);
}
/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_DEBUG_DICT_H */
