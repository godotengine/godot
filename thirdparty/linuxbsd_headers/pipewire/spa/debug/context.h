/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2023 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_DEBUG_CONTEXT_H
#define SPA_DEBUG_CONTEXT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdarg.h>

#include <spa/utils/defs.h>
/**
 * \addtogroup spa_debug
 * \{
 */

#ifndef spa_debugn
#define spa_debugn(_fmt,...)	printf((_fmt), ## __VA_ARGS__)
#endif
#ifndef spa_debug
#define spa_debug(_fmt,...)	spa_debugn(_fmt"\n", ## __VA_ARGS__)
#endif

struct spa_debug_context {
	void (*log) (struct spa_debug_context *ctx, const char *fmt, ...) SPA_PRINTF_FUNC(2, 3);
};

#define spa_debugc(_c,_fmt,...)	(_c)?((_c)->log((_c),_fmt, ## __VA_ARGS__)):(void)spa_debug(_fmt, ## __VA_ARGS__)

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_DEBUG_CONTEXT_H */
