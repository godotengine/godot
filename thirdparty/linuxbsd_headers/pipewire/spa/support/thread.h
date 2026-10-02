/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2021 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_THREAD_H
#define SPA_THREAD_H

#ifdef __cplusplus
extern "C" {
#endif

#include <string.h>
#include <errno.h>

#include <spa/utils/defs.h>
#include <spa/utils/hook.h>
#include <spa/utils/dict.h>

/** \defgroup spa_thread Thread
 * Threading utility interfaces
 */

/**
 * \addtogroup spa_thread
 * \{
 */

/** a thread object.
 * This can be cast to a platform native thread, like pthread on posix systems
 */
#define SPA_TYPE_INFO_Thread		SPA_TYPE_INFO_POINTER_BASE "Thread"
struct spa_thread;

#define SPA_TYPE_INTERFACE_ThreadUtils	SPA_TYPE_INFO_INTERFACE_BASE "ThreadUtils"
#define SPA_VERSION_THREAD_UTILS		0
struct spa_thread_utils { struct spa_interface iface; };

/** thread utils */
struct spa_thread_utils_methods {
#define SPA_VERSION_THREAD_UTILS_METHODS	0
	uint32_t version;

	/** create a new thread that runs \a start with \a arg */
	struct spa_thread * (*create) (void *object, const struct spa_dict *props,
			void *(*start)(void*), void *arg);
	/** stop and join a thread */
	int (*join)(void *object, struct spa_thread *thread, void **retval);

	/** get realtime priority range for threads created with \a props */
	int (*get_rt_range) (void *object, const struct spa_dict *props, int *min, int *max);
	/** acquire realtime priority, a priority of -1 refers to the priority
	 * configured in the realtime module
	 */
	int (*acquire_rt) (void *object, struct spa_thread *thread, int priority);
	/** drop realtime priority */
	int (*drop_rt) (void *object, struct spa_thread *thread);
};

/** \copydoc spa_thread_utils_methods.create
 * \sa spa_thread_utils_methods.create */
static inline struct spa_thread *spa_thread_utils_create(struct spa_thread_utils *o,
		const struct spa_dict *props, void *(*start_routine)(void*), void *arg)
{
	struct spa_thread *res = NULL;
	spa_interface_call_res(&o->iface,
			struct spa_thread_utils_methods, res, create, 0,
			props, start_routine, arg);
	return res;
}

/** \copydoc spa_thread_utils_methods.join
 * \sa spa_thread_utils_methods.join */
static inline int spa_thread_utils_join(struct spa_thread_utils *o,
		struct spa_thread *thread, void **retval)
{
	int res = -ENOTSUP;
	spa_interface_call_res(&o->iface,
			struct spa_thread_utils_methods, res, join, 0,
			thread, retval);
	return res;
}

/** \copydoc spa_thread_utils_methods.get_rt_range
 * \sa spa_thread_utils_methods.get_rt_range */
static inline int spa_thread_utils_get_rt_range(struct spa_thread_utils *o,
		const struct spa_dict *props, int *min, int *max)
{
	int res = -ENOTSUP;
	spa_interface_call_res(&o->iface,
			struct spa_thread_utils_methods, res, get_rt_range, 0,
			props, min, max);
	return res;
}

/** \copydoc spa_thread_utils_methods.acquire_rt
 * \sa spa_thread_utils_methods.acquire_rt */
static inline int spa_thread_utils_acquire_rt(struct spa_thread_utils *o,
		struct spa_thread *thread, int priority)
{
	int res = -ENOTSUP;
	spa_interface_call_res(&o->iface,
			struct spa_thread_utils_methods, res, acquire_rt, 0,
			thread, priority);
	return res;
}

/** \copydoc spa_thread_utils_methods.drop_rt
 * \sa spa_thread_utils_methods.drop_rt */
static inline int spa_thread_utils_drop_rt(struct spa_thread_utils *o,
		struct spa_thread *thread)
{
	int res = -ENOTSUP;
	spa_interface_call_res(&o->iface,
			struct spa_thread_utils_methods, res, drop_rt, 0, thread);
	return res;
}

#define SPA_KEY_THREAD_NAME		"thread.name"		/* the thread name */
#define SPA_KEY_THREAD_STACK_SIZE	"thread.stack-size"	/* the stack size of the thread */

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_THREAD_H */
