/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_LOOP_H
#define SPA_LOOP_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/utils/defs.h>
#include <spa/utils/hook.h>
#include <spa/support/system.h>

/** \defgroup spa_loop Loop
 * Event loop interface
 */

/**
 * \addtogroup spa_loop
 * \{
 */

#define SPA_TYPE_INTERFACE_Loop		SPA_TYPE_INFO_INTERFACE_BASE "Loop"
#define SPA_TYPE_INTERFACE_DataLoop	SPA_TYPE_INFO_INTERFACE_BASE "DataLoop"
#define SPA_VERSION_LOOP		0
struct spa_loop { struct spa_interface iface; };

#define SPA_TYPE_INTERFACE_LoopControl	SPA_TYPE_INFO_INTERFACE_BASE "LoopControl"
#define SPA_VERSION_LOOP_CONTROL	1
struct spa_loop_control { struct spa_interface iface; };

#define SPA_TYPE_INTERFACE_LoopUtils	SPA_TYPE_INFO_INTERFACE_BASE "LoopUtils"
#define SPA_VERSION_LOOP_UTILS		0
struct spa_loop_utils { struct spa_interface iface; };

struct spa_source;

typedef void (*spa_source_func_t) (struct spa_source *source);

struct spa_source {
	struct spa_loop *loop;
	spa_source_func_t func;
	void *data;
	int fd;
	uint32_t mask;
	uint32_t rmask;
	/* private data for the loop implementer */
	void *priv;
};

typedef int (*spa_invoke_func_t) (struct spa_loop *loop,
				  bool async,
				  uint32_t seq,
				  const void *data,
				  size_t size,
				  void *user_data);

/**
 * Register sources and work items to an event loop
 */
struct spa_loop_methods {
	/* the version of this structure. This can be used to expand this
	 * structure in the future */
#define SPA_VERSION_LOOP_METHODS	0
	uint32_t version;

	/** add a source to the loop */
	int (*add_source) (void *object,
			   struct spa_source *source);

	/** update the source io mask */
	int (*update_source) (void *object,
			struct spa_source *source);

	/** remove a source from the loop */
	int (*remove_source) (void *object,
			struct spa_source *source);

	/** invoke a function in the context of this loop */
	int (*invoke) (void *object,
		       spa_invoke_func_t func,
		       uint32_t seq,
		       const void *data,
		       size_t size,
		       bool block,
		       void *user_data);
};

#define spa_loop_method(o,method,version,...)				\
({									\
	int _res = -ENOTSUP;						\
	struct spa_loop *_o = o;					\
	spa_interface_call_res(&_o->iface,				\
			struct spa_loop_methods, _res,			\
			method, version, ##__VA_ARGS__);		\
	_res;								\
})

#define spa_loop_add_source(l,...)	spa_loop_method(l,add_source,0,##__VA_ARGS__)
#define spa_loop_update_source(l,...)	spa_loop_method(l,update_source,0,##__VA_ARGS__)
#define spa_loop_remove_source(l,...)	spa_loop_method(l,remove_source,0,##__VA_ARGS__)
#define spa_loop_invoke(l,...)		spa_loop_method(l,invoke,0,##__VA_ARGS__)


/** Control hooks. These hooks can't be removed from their
 *  callbacks and must be removed from a safe place (when the loop
 *  is not running or when it is locked). */
struct spa_loop_control_hooks {
#define SPA_VERSION_LOOP_CONTROL_HOOKS	0
	uint32_t version;
	/** Executed right before waiting for events. It is typically used to
	 * release locks. */
	void (*before) (void *data);
	/** Executed right after waiting for events. It is typically used to
	 * reacquire locks. */
	void (*after) (void *data);
};

#define spa_loop_control_hook_before(l)							\
({											\
	struct spa_hook_list *_l = l;							\
	struct spa_hook *_h;								\
	spa_list_for_each_reverse(_h, &_l->list, link)					\
		spa_callbacks_call_fast(&_h->cb, struct spa_loop_control_hooks, before, 0);	\
})

#define spa_loop_control_hook_after(l)							\
({											\
	struct spa_hook_list *_l = l;							\
	struct spa_hook *_h;								\
	spa_list_for_each(_h, &_l->list, link)						\
		spa_callbacks_call_fast(&_h->cb, struct spa_loop_control_hooks, after, 0);	\
})

/**
 * Control an event loop
 */
struct spa_loop_control_methods {
	/* the version of this structure. This can be used to expand this
	 * structure in the future */
#define SPA_VERSION_LOOP_CONTROL_METHODS	1
	uint32_t version;

	int (*get_fd) (void *object);

	/** Add a hook
	 * \param ctrl the control to change
	 * \param hooks the hooks to add
	 *
	 * Adds hooks to the loop controlled by \a ctrl.
	 */
	void (*add_hook) (void *object,
			  struct spa_hook *hook,
			  const struct spa_loop_control_hooks *hooks,
			  void *data);

	/** Enter a loop
	 * \param ctrl the control
	 *
	 * Start an iteration of the loop. This function should be called
	 * before calling iterate and is typically used to capture the thread
	 * that this loop will run in.
	 */
	void (*enter) (void *object);
	/** Leave a loop
	 * \param ctrl the control
	 *
	 * Ends the iteration of a loop. This should be called after calling
	 * iterate.
	 */
	void (*leave) (void *object);

	/** Perform one iteration of the loop.
	 * \param ctrl the control
	 * \param timeout an optional timeout in milliseconds.
	 *	0 for no timeout, -1 for infinite timeout.
	 *
	 * This function will block
	 * up to \a timeout milliseconds and then dispatch the fds with activity.
	 * The number of dispatched fds is returned.
	 */
	int (*iterate) (void *object, int timeout);

	/** Check context of the loop
	 * \param ctrl the control
	 *
	 * This function will check if the current thread is currently the
	 * one that did the enter call. Since version 1:1.
	 *
	 * returns 1 on success, 0 or negative errno value on error.
	 */
	int (*check) (void *object);
};

#define spa_loop_control_method_v(o,method,version,...)			\
({									\
	struct spa_loop_control *_o = o;				\
	spa_interface_call(&_o->iface,					\
			struct spa_loop_control_methods,		\
			method, version, ##__VA_ARGS__);		\
})

#define spa_loop_control_method_r(o,method,version,...)			\
({									\
	int _res = -ENOTSUP;						\
	struct spa_loop_control *_o = o;				\
	spa_interface_call_res(&_o->iface,				\
			struct spa_loop_control_methods, _res,		\
			method, version, ##__VA_ARGS__);		\
	_res;								\
})

#define spa_loop_control_method_fast_r(o,method,version,...)		\
({									\
	int _res;							\
	struct spa_loop_control *_o = o;				\
	spa_interface_call_fast_res(&_o->iface,				\
			struct spa_loop_control_methods, _res,		\
			method, version, ##__VA_ARGS__);		\
	_res;								\
})

#define spa_loop_control_get_fd(l)		spa_loop_control_method_r(l,get_fd,0)
#define spa_loop_control_add_hook(l,...)	spa_loop_control_method_v(l,add_hook,0,__VA_ARGS__)
#define spa_loop_control_enter(l)		spa_loop_control_method_v(l,enter,0)
#define spa_loop_control_leave(l)		spa_loop_control_method_v(l,leave,0)
#define spa_loop_control_iterate(l,...)		spa_loop_control_method_r(l,iterate,0,__VA_ARGS__)
#define spa_loop_control_check(l)		spa_loop_control_method_r(l,check,1)

#define spa_loop_control_iterate_fast(l,...)	spa_loop_control_method_fast_r(l,iterate,0,__VA_ARGS__)

typedef void (*spa_source_io_func_t) (void *data, int fd, uint32_t mask);
typedef void (*spa_source_idle_func_t) (void *data);
typedef void (*spa_source_event_func_t) (void *data, uint64_t count);
typedef void (*spa_source_timer_func_t) (void *data, uint64_t expirations);
typedef void (*spa_source_signal_func_t) (void *data, int signal_number);

/**
 * Create sources for an event loop
 */
struct spa_loop_utils_methods {
	/* the version of this structure. This can be used to expand this
	 * structure in the future */
#define SPA_VERSION_LOOP_UTILS_METHODS	0
	uint32_t version;

	struct spa_source *(*add_io) (void *object,
				      int fd,
				      uint32_t mask,
				      bool close,
				      spa_source_io_func_t func, void *data);

	int (*update_io) (void *object, struct spa_source *source, uint32_t mask);

	struct spa_source *(*add_idle) (void *object,
					bool enabled,
					spa_source_idle_func_t func, void *data);
	int (*enable_idle) (void *object, struct spa_source *source, bool enabled);

	struct spa_source *(*add_event) (void *object,
					 spa_source_event_func_t func, void *data);
	int (*signal_event) (void *object, struct spa_source *source);

	struct spa_source *(*add_timer) (void *object,
					 spa_source_timer_func_t func, void *data);
	int (*update_timer) (void *object,
			     struct spa_source *source,
			     struct timespec *value,
			     struct timespec *interval,
			     bool absolute);
	struct spa_source *(*add_signal) (void *object,
					  int signal_number,
					  spa_source_signal_func_t func, void *data);

	/** destroy a source allocated with this interface. This function
	 * should only be called when the loop is not running or from the
	 * context of the running loop */
	void (*destroy_source) (void *object, struct spa_source *source);
};

#define spa_loop_utils_method_v(o,method,version,...)			\
({									\
	struct spa_loop_utils *_o = o;					\
	spa_interface_call(&_o->iface,					\
			struct spa_loop_utils_methods,			\
			method, version, ##__VA_ARGS__);		\
})

#define spa_loop_utils_method_r(o,method,version,...)			\
({									\
	int _res = -ENOTSUP;						\
	struct spa_loop_utils *_o = o;					\
	spa_interface_call_res(&_o->iface,				\
			struct spa_loop_utils_methods, _res,		\
			method, version, ##__VA_ARGS__);		\
	_res;								\
})
#define spa_loop_utils_method_s(o,method,version,...)			\
({									\
	struct spa_source *_res = NULL;					\
	struct spa_loop_utils *_o = o;					\
	spa_interface_call_res(&_o->iface,				\
			struct spa_loop_utils_methods, _res,		\
			method, version, ##__VA_ARGS__);		\
	_res;								\
})


#define spa_loop_utils_add_io(l,...)		spa_loop_utils_method_s(l,add_io,0,__VA_ARGS__)
#define spa_loop_utils_update_io(l,...)		spa_loop_utils_method_r(l,update_io,0,__VA_ARGS__)
#define spa_loop_utils_add_idle(l,...)		spa_loop_utils_method_s(l,add_idle,0,__VA_ARGS__)
#define spa_loop_utils_enable_idle(l,...)	spa_loop_utils_method_r(l,enable_idle,0,__VA_ARGS__)
#define spa_loop_utils_add_event(l,...)		spa_loop_utils_method_s(l,add_event,0,__VA_ARGS__)
#define spa_loop_utils_signal_event(l,...)	spa_loop_utils_method_r(l,signal_event,0,__VA_ARGS__)
#define spa_loop_utils_add_timer(l,...)		spa_loop_utils_method_s(l,add_timer,0,__VA_ARGS__)
#define spa_loop_utils_update_timer(l,...)	spa_loop_utils_method_r(l,update_timer,0,__VA_ARGS__)
#define spa_loop_utils_add_signal(l,...)	spa_loop_utils_method_s(l,add_signal,0,__VA_ARGS__)
#define spa_loop_utils_destroy_source(l,...)	spa_loop_utils_method_v(l,destroy_source,0,__VA_ARGS__)

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_LOOP_H */
