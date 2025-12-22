/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2023 PipeWire authors */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_UTILS_CLEANUP_H
#define SPA_UTILS_CLEANUP_H

#define spa_exchange(var, new_value) \
__extension__ ({ \
	__typeof__(var) *_ptr = &(var); \
	__typeof__(var) _old_value = *_ptr; \
	*_ptr = (new_value); \
	_old_value; \
})

/* ========================================================================== */

#if __GNUC__ >= 10 || defined(__clang__)
#define spa_steal_ptr(ptr) ((__typeof__(*(ptr)) *) spa_exchange((ptr), NULL))
#else
#define spa_steal_ptr(ptr) spa_exchange((ptr), NULL)
#endif

#define spa_clear_ptr(ptr, destructor) \
__extension__ ({ \
	__typeof__(ptr) _old_value = spa_steal_ptr(ptr); \
	if (_old_value) \
		destructor(_old_value); \
	(void) 0; \
})

/* ========================================================================== */

#include <unistd.h>

#define spa_steal_fd(fd) spa_exchange((fd), -1)

#define spa_clear_fd(fd) \
__extension__ ({ \
	int _old_value = spa_steal_fd(fd), _res = 0; \
	if (_old_value >= 0) \
		_res = close(_old_value); \
	_res; \
})

/* ========================================================================== */

#if defined(__has_attribute) && __has_attribute(__cleanup__)

#define spa_cleanup(func) __attribute__((__cleanup__(func)))

#define SPA_DEFINE_AUTO_CLEANUP(name, type, ...) \
typedef __typeof__(type) _spa_auto_cleanup_type_ ## name; \
static inline void _spa_auto_cleanup_func_ ## name (__typeof__(type) *thing) \
{ \
	__VA_ARGS__ \
}

#define spa_auto(name) \
	spa_cleanup(_spa_auto_cleanup_func_ ## name) \
	_spa_auto_cleanup_type_ ## name

#define SPA_DEFINE_AUTOPTR_CLEANUP(name, type, ...) \
typedef __typeof__(type) * _spa_autoptr_cleanup_type_ ## name; \
static inline void _spa_autoptr_cleanup_func_ ## name (__typeof__(type) **thing) \
{ \
	__VA_ARGS__ \
}

#define spa_autoptr(name) \
	spa_cleanup(_spa_autoptr_cleanup_func_ ## name) \
	_spa_autoptr_cleanup_type_ ## name

/* ========================================================================== */

#include <stdlib.h>

static inline void _spa_autofree_cleanup_func(void *p)
{
	free(*(void **) p);
}
#define spa_autofree spa_cleanup(_spa_autofree_cleanup_func)

/* ========================================================================== */

static inline void _spa_autoclose_cleanup_func(int *fd)
{
	spa_clear_fd(*fd);
}
#define spa_autoclose spa_cleanup(_spa_autoclose_cleanup_func)

/* ========================================================================== */

#include <stdio.h>

SPA_DEFINE_AUTOPTR_CLEANUP(FILE, FILE, {
	spa_clear_ptr(*thing, fclose);
})

/* ========================================================================== */

#include <dirent.h>

SPA_DEFINE_AUTOPTR_CLEANUP(DIR, DIR, {
	spa_clear_ptr(*thing, closedir);
})

#else

#define SPA_DEFINE_AUTO_CLEANUP(name, type, ...)
#define SPA_DEFINE_AUTOPTR_CLEANUP(name, type, ...)

#endif

#endif /* SPA_UTILS_CLEANUP_H */
