/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_HOOK_H
#define SPA_HOOK_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/utils/defs.h>
#include <spa/utils/list.h>

/** \defgroup spa_interfaces Interfaces
 *
 * \brief Generic implementation of implementation-independent interfaces
 *
 * A SPA Interface is a generic struct that, together with a few macros,
 * provides a generic way of invoking methods on objects without knowing the
 * details of the implementation.
 *
 * The primary interaction with interfaces is through macros that expand into
 * the right method call. For the implementation of an interface, we need two
 * structs and a macro to invoke the `bar` method:
 *
 * \code{.c}
 * // this struct must be public and defines the interface to a
 * // struct foo
 * struct foo_methods {
 *     uint32_t version;
 *     void (*bar)(void *object, const char *msg);
 * };
 *
 * // this struct does not need to be public
 * struct foo {
 *     struct spa_interface iface; // must be first element, see foo_bar()
 *     int some_other_field;
 *     ...
 * };
 *
 * // if struct foo is private, we need to cast to a
 * // generic spa_interface object
 * #define foo_bar(obj, ...) ({ \
 *     struct foo *f = obj;
 *     spa_interface_call((struct spa_interface *)f, // pointer to spa_interface in foo
 *                        struct foo_methods, // type of callbacks
 *                        bar, // name of methods
 *                        0, // hardcoded version to match foo_methods->version
 *                        __VA_ARGS__ // pass rest of args through
 *                        );/
 * })
 * \endcode
 *
 * The `struct foo_methods` and the invocation macro `foo_bar()` must be
 * available to the caller. The implementation of `struct foo` can be private.
 *
 * \code{.c}
 * void main(void) {
 *      struct foo *myfoo = get_foo_from_somewhere();
 *      foo_bar(myfoo, "Invoking bar() on myfoo");
 * }
 * \endcode
 * The expansion of `foo_bar()` resolves roughly into this code:
 * \code{.c}
 * void main(void) {
 *     struct foo *myfoo = get_foo_from_somewhere();
 *     // foo_bar(myfoo, "Invoking bar() on myfoo");
 *     const struct foo_methods *methods = ((struct spa_interface*)myfoo)->cb;
 *     if (0 >= methods->version && // version check
 *         methods->bar) // compile error if this function does not exist,
 *             methods->bar(myfoo, "Invoking bar() on myfoo");
 * }
 * \endcode
 *
 * The typecast used in `foo_bar()` allows `struct foo` to be opaque to the
 * caller. The implementation may assign the callback methods at object
 * instantiation, and the caller will transparently invoke the method on the
 * given object. For example, the following code assigns a different `bar()` method on
 * Mondays - the caller does not need to know this.
 * \code{.c}
 *
 * static void bar_stdout(struct foo *f, const char *msg) {
 *     printf(msg);
 * }
 * static void bar_stderr(struct foo *f, const char *msg) {
 *     fprintf(stderr, msg);
 * }
 *
 * struct foo* get_foo_from_somewhere() {
 *     struct foo *f = calloc(sizeof struct foo);
 *     // illustrative only, use SPA_INTERFACE_INIT()
 *     f->iface->cb = (struct foo_methods*) { .bar = bar_stdout };
 *     if (today_is_monday)
 *         f->iface->cb = (struct foo_methods*) { .bar = bar_stderr };
 *     return f;
 * }
 * \endcode
 */

/**
 * \addtogroup spa_interfaces
 * \{
 */

/** \struct spa_callbacks
 * Callbacks, contains the structure with functions and the data passed
 * to the functions.  The structure should also contain a version field that
 * is checked. */
struct spa_callbacks {
	const void *funcs;
	void *data;
};

/** Check if a callback \a c is of at least version \a v */
#define SPA_CALLBACK_VERSION_MIN(c,v) ((c) && ((v) == 0 || (c)->version > (v)-1))

/** Check if a callback \a c has method \a m of version \a v */
#define SPA_CALLBACK_CHECK(c,m,v) (SPA_CALLBACK_VERSION_MIN(c,v) && (c)->m)

/**
 * Initialize the set of functions \a funcs as a \ref spa_callbacks, together
 * with \a _data.
 */
#define SPA_CALLBACKS_INIT(_funcs,_data) ((struct spa_callbacks){ (_funcs), (_data), })

/** \struct spa_interface
 */
struct spa_interface {
	const char *type;
	uint32_t version;
	struct spa_callbacks cb;
};

/**
 * Initialize a \ref spa_interface.
 *
 * \code{.c}
 * const static struct foo_methods foo_funcs = {
 *    .bar = some_bar_implementation,
 * };
 *
 * struct foo *f = malloc(...);
 * f->iface = SPA_INTERFACE_INIT("foo type", 0, foo_funcs, NULL);
 * \endcode
 *
 */
#define SPA_INTERFACE_INIT(_type,_version,_funcs,_data) \
	((struct spa_interface){ (_type), (_version), SPA_CALLBACKS_INIT(_funcs,_data), })

/**
 * Invoke method named \a method in the \a callbacks.
 * The \a method_type defines the type of the method struct.
 * Returns true if the method could be called, false otherwise.
 */
#define spa_callbacks_call(callbacks,type,method,vers,...)			\
({										\
	const type *_f = (const type *) (callbacks)->funcs;			\
	bool _res = SPA_CALLBACK_CHECK(_f,method,vers);				\
	if (SPA_LIKELY(_res))							\
		_f->method((callbacks)->data, ## __VA_ARGS__);			\
	_res;									\
})

#define spa_callbacks_call_fast(callbacks,type,method,vers,...)			\
({										\
	const type *_f = (const type *) (callbacks)->funcs;			\
	_f->method((callbacks)->data, ## __VA_ARGS__);				\
	true;									\
})


/**
 * True if the \a callbacks are of version \a vers, false otherwise
 */
#define spa_callback_version_min(callbacks,type,vers)				\
({										\
	const type *_f = (const type *) (callbacks)->funcs;			\
	SPA_CALLBACK_VERSION_MIN(_f,vers);					\
})

/**
 * True if the \a callbacks contains \a method of version
 * \a vers, false otherwise
 */
#define spa_callback_check(callbacks,type,method,vers)				\
({										\
	const type *_f = (const type *) (callbacks)->funcs;			\
	SPA_CALLBACK_CHECK(_f,method,vers);					\
})

/**
 * Invoke method named \a method in the \a callbacks.
 * The \a method_type defines the type of the method struct.
 *
 * The return value is stored in \a res.
 */
#define spa_callbacks_call_res(callbacks,type,res,method,vers,...)		\
({										\
	const type *_f = (const type *) (callbacks)->funcs;			\
	if (SPA_LIKELY(SPA_CALLBACK_CHECK(_f,method,vers)))			\
		res = _f->method((callbacks)->data, ## __VA_ARGS__);		\
	res;									\
})
#define spa_callbacks_call_fast_res(callbacks,type,res,method,vers,...)		\
({										\
	const type *_f = (const type *) (callbacks)->funcs;			\
	res = _f->method((callbacks)->data, ## __VA_ARGS__);			\
})

/**
 * True if the \a iface's callbacks are of version \a vers, false otherwise
 */
#define spa_interface_callback_version_min(iface,method_type,vers)		\
   spa_callback_version_min(&(iface)->cb, method_type, vers)

/**
 * True if the \a iface's callback \a method is of version \a vers
 * and exists, false otherwise
 */
#define spa_interface_callback_check(iface,method_type,method,vers)		\
   spa_callback_check(&(iface)->cb, method_type, method, vers)

/**
 * Invoke method named \a method in the callbacks on the given interface object.
 * The \a method_type defines the type of the method struct, not the interface
 * itself.
 */
#define spa_interface_call(iface,method_type,method,vers,...)			\
	spa_callbacks_call(&(iface)->cb,method_type,method,vers,##__VA_ARGS__)

#define spa_interface_call_fast(iface,method_type,method,vers,...)		\
	spa_callbacks_call_fast(&(iface)->cb,method_type,method,vers,##__VA_ARGS__)

/**
 * Invoke method named \a method in the callbacks on the given interface object.
 * The \a method_type defines the type of the method struct, not the interface
 * itself.
 *
 * The return value is stored in \a res.
 */
#define spa_interface_call_res(iface,method_type,res,method,vers,...)			\
	spa_callbacks_call_res(&(iface)->cb,method_type,res,method,vers,##__VA_ARGS__)

#define spa_interface_call_fast_res(iface,method_type,res,method,vers,...)		\
	spa_callbacks_call_fast_res(&(iface)->cb,method_type,res,method,vers,##__VA_ARGS__)

/**
 * \}
 */

/** \defgroup spa_hooks Hooks
 *
 * A SPA Hook is a data structure to keep track of callbacks. It is similar to
 * the \ref spa_interfaces and typically used where an implementation allows
 * for multiple external callback functions. For example, an implementation may
 * use a hook list to implement signals with each caller using a hook to
 * register callbacks to be invoked on those signals.
 *
 * The below (pseudo)code is a minimal example outlining the use of hooks:
 * \code{.c}
 * // the public interface
 * #define VERSION_BAR_EVENTS 0 // version of the vtable
 * struct bar_events {
 *    uint32_t version; // NOTE: an integral member named `version`
 *                      //       must be present in the vtable
 *    void (*boom)(void *data, const char *msg);
 * };
 *
 * // private implementation
 * struct party {
 *     struct spa_hook_list bar_list;
 * };
 *
 * void party_add_event_listener(struct party *p, struct spa_hook *listener,
 *                               const struct bar_events *events, void *data)
 * {
 *    spa_hook_list_append(&p->bar_list, listener, events, data);
 * }
 *
 * static void party_on(struct party *p)
 * {
 *     // NOTE: this is a macro, it evaluates to an integer,
 *     //       which is the number of hooks called
 *     spa_hook_list_call(&p->list,
 *                        struct bar_events, // vtable type
 *                        boom,              // function name
 *                        0,                 // hardcoded version,
 *                                           //     usually the version in which `boom`
 *                                           //     has been added to the vtable
 *                        "party on, wayne"  // function argument(s)
 *                        );
 * }
 * \endcode
 *
 * In the caller, the hooks can be used like this:
 * \code{.c}
 * static void boom_cb(void *data, const char *msg) {
 *      // data is userdata from main()
 *      printf("%s", msg);
 * }
 *
 * static const struct bar_events events = {
 *    .version = VERSION_BAR_EVENTS, // version of the implemented interface
 *    .boom = boom_cb,
 * };
 *
 * void main(void) {
 *      void *userdata = whatever;
 *      struct spa_hook hook;
 *      struct party *p = start_the_party();
 *
 *      party_add_event_listener(p, &hook, &events, userdata);
 *
 *      mainloop();
 *      return 0;
 * }
 *
 * \endcode
 */

/**
 * \addtogroup spa_hooks
 * \{
 */

/** \struct spa_hook_list
 * A list of hooks. This struct is primarily used by
 * implementation that use multiple caller-provided \ref spa_hook. */
struct spa_hook_list {
	struct spa_list list;
};


/** \struct spa_hook
 * A hook, contains the structure with functions and the data passed
 * to the functions.
 *
 * A hook should be treated as opaque by the caller.
 */
struct spa_hook {
	struct spa_list link;
	struct spa_callbacks cb;
	/** callback and data for the hook list, private to the
	  * hook_list implementor */
	void (*removed) (struct spa_hook *hook);
	void *priv;
};

/** Initialize a hook list to the empty list*/
static inline void spa_hook_list_init(struct spa_hook_list *list)
{
	spa_list_init(&list->list);
}

static inline bool spa_hook_list_is_empty(struct spa_hook_list *list)
{
	return spa_list_is_empty(&list->list);
}

/** Append a hook. */
static inline void spa_hook_list_append(struct spa_hook_list *list,
					struct spa_hook *hook,
					const void *funcs, void *data)
{
	spa_zero(*hook);
	hook->cb = SPA_CALLBACKS_INIT(funcs, data);
	spa_list_append(&list->list, &hook->link);
}

/** Prepend a hook */
static inline void spa_hook_list_prepend(struct spa_hook_list *list,
					 struct spa_hook *hook,
					 const void *funcs, void *data)
{
	spa_zero(*hook);
	hook->cb = SPA_CALLBACKS_INIT(funcs, data);
	spa_list_prepend(&list->list, &hook->link);
}

/** Remove a hook */
static inline void spa_hook_remove(struct spa_hook *hook)
{
	if (spa_list_is_initialized(&hook->link))
		spa_list_remove(&hook->link);
	if (hook->removed)
		hook->removed(hook);
}

/** Remove all hooks from the list */
static inline void spa_hook_list_clean(struct spa_hook_list *list)
{
	struct spa_hook *h;
	spa_list_consume(h, &list->list, link)
		spa_hook_remove(h);
}

static inline void
spa_hook_list_isolate(struct spa_hook_list *list,
		struct spa_hook_list *save,
		struct spa_hook *hook,
		const void *funcs, void *data)
{
	/* init save list and move hooks to it */
	spa_hook_list_init(save);
	spa_list_insert_list(&save->list, &list->list);
	/* init hooks and add single hook */
	spa_hook_list_init(list);
	spa_hook_list_append(list, hook, funcs, data);
}

static inline void
spa_hook_list_join(struct spa_hook_list *list,
		struct spa_hook_list *save)
{
	spa_list_insert_list(&list->list, &save->list);
}

#define spa_hook_list_call_simple(l,type,method,vers,...)			\
({										\
	struct spa_hook_list *_l = l;						\
	struct spa_hook *_h, *_t;						\
	spa_list_for_each_safe(_h, _t, &_l->list, link)				\
		spa_callbacks_call(&_h->cb,type,method,vers, ## __VA_ARGS__);	\
})

/** Call all hooks in a list, starting from the given one and optionally stopping
 * after calling the first non-NULL function, returns the number of methods
 * called */
#define spa_hook_list_do_call(l,start,type,method,vers,once,...)		\
({										\
	struct spa_hook_list *_list = l;					\
	struct spa_list *_s = start ? (struct spa_list *)start : &_list->list;	\
	struct spa_hook _cursor = { 0 }, *_ci;					\
	int _count = 0;								\
	spa_list_cursor_start(_cursor, _s, link);				\
	spa_list_for_each_cursor(_ci, _cursor, &_list->list, link) {		\
		if (spa_callbacks_call(&_ci->cb,type,method,vers, ## __VA_ARGS__)) {		\
			_count++;						\
			if (once)						\
				break;						\
		}								\
	}									\
	spa_list_cursor_end(_cursor, link);					\
	_count;									\
})

/**
 * Call the method named \a m for each element in list \a l.
 * \a t specifies the type of the callback struct.
 */
#define spa_hook_list_call(l,t,m,v,...)			spa_hook_list_do_call(l,NULL,t,m,v,false,##__VA_ARGS__)
/**
 * Call the method named \a m for each element in list \a l, stopping after
 * the first invocation.
 * \a t specifies the type of the callback struct.
 */
#define spa_hook_list_call_once(l,t,m,v,...)		spa_hook_list_do_call(l,NULL,t,m,v,true,##__VA_ARGS__)

#define spa_hook_list_call_start(l,s,t,m,v,...)		spa_hook_list_do_call(l,s,t,m,v,false,##__VA_ARGS__)
#define spa_hook_list_call_once_start(l,s,t,m,v,...)	spa_hook_list_do_call(l,s,t,m,v,true,##__VA_ARGS__)

/**
 * \}
 */

#ifdef __cplusplus
}
#endif

#endif /* SPA_HOOK_H */
