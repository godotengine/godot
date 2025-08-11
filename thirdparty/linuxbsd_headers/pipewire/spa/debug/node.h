/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_DEBUG_NODE_H
#define SPA_DEBUG_NODE_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \addtogroup spa_debug
 * \{
 */

#include <spa/node/node.h>
#include <spa/debug/context.h>
#include <spa/debug/dict.h>

static inline int spa_debugc_port_info(struct spa_debug_context *ctx, int indent, const struct spa_port_info *info)
{
        spa_debugc(ctx, "%*s" "struct spa_port_info %p:", indent, "", info);
        spa_debugc(ctx, "%*s" " flags: \t%08" PRIx64, indent, "", info->flags);
        spa_debugc(ctx, "%*s" " rate: \t%d/%d", indent, "", info->rate.num, info->rate.denom);
        spa_debugc(ctx, "%*s" " props:", indent, "");
        if (info->props)
                spa_debugc_dict(ctx, indent + 2, info->props);
        else
                spa_debugc(ctx, "%*s" "  none", indent, "");
        return 0;
}

static inline int spa_debug_port_info(int indent, const struct spa_port_info *info)
{
	return spa_debugc_port_info(NULL, indent, info);
}
/**
 * \}
 */


#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_DEBUG_NODE_H */
