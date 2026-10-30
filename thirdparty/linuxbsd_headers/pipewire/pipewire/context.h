/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_CONTEXT_H
#define PIPEWIRE_CONTEXT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/utils/defs.h>
#include <spa/utils/hook.h>

/** \defgroup pw_context Context
 *
 * \brief The PipeWire context object manages all locally available
 * resources. It is used by both clients and servers.
 *
 * The context is used to:
 *
 *  - Load modules and extend the functionality. This includes
 *    extending the protocol with new object types or creating
 *    any of the available objects.
 *
 *  - Create implementations of various objects like nodes,
 *    devices, factories, modules, etc.. This will usually also
 *    create pw_global objects that can then be shared with
 *    clients.
 *
 *  - Connect to another PipeWire instance (the main daemon, for
 *    example) and interact with it (See \ref page_core_api).
 *
 *  - Export a local implementation of an object to another
 *    instance.
 */

/**
 * \addtogroup pw_context
 * @{
 */
struct pw_context;

struct pw_global;
struct pw_impl_client;
struct pw_impl_node;

#include <pipewire/core.h>
#include <pipewire/loop.h>
#include <pipewire/properties.h>

/** context events emitted by the context object added with \ref pw_context_add_listener */
struct pw_context_events {
#define PW_VERSION_CONTEXT_EVENTS	1
	uint32_t version;

	/** The context is being destroyed */
	void (*destroy) (void *data);
	/** The context is being freed */
	void (*free) (void *data);
	/** a new client object is added */
	void (*check_access) (void *data, struct pw_impl_client *client);
	/** a new global object was added */
	void (*global_added) (void *data, struct pw_global *global);
	/** a global object was removed */
	void (*global_removed) (void *data, struct pw_global *global);

	/** a driver was added, since 0.3.75 version:1 */
	void (*driver_added) (void *data, struct pw_impl_node *node);
	/** a driver was removed, since 0.3.75 version:1 */
	void (*driver_removed) (void *data, struct pw_impl_node *node);
};

/** Make a new context object for a given main_loop. Ownership of the properties is taken */
struct pw_context * pw_context_new(struct pw_loop *main_loop,		/**< a main loop to run in */
			     struct pw_properties *props,	/**< extra properties */
			     size_t user_data_size		/**< extra user data size */);

/** destroy a context object, all resources except the main_loop will be destroyed */
void pw_context_destroy(struct pw_context *context);

/** Get the context user data */
void *pw_context_get_user_data(struct pw_context *context);

/** Add a new event listener to a context */
void pw_context_add_listener(struct pw_context *context,
			  struct spa_hook *listener,
			  const struct pw_context_events *events,
			  void *data);

/** Get the context properties */
const struct pw_properties *pw_context_get_properties(struct pw_context *context);

/** Update the context properties */
int pw_context_update_properties(struct pw_context *context, const struct spa_dict *dict);

/** Get a config section for this context. Since 0.3.22, deprecated,
 * use pw_context_conf_section_for_each(). */
const char *pw_context_get_conf_section(struct pw_context *context, const char *section);
/** Parse a standard config section for this context. Since 0.3.22 */
int pw_context_parse_conf_section(struct pw_context *context,
		struct pw_properties *conf, const char *section);

/** update properties from a section into props. Since 0.3.45 */
int pw_context_conf_update_props(struct pw_context *context, const char *section,
		struct pw_properties *props);
/** emit callback for all config sections. Since 0.3.45 */
int pw_context_conf_section_for_each(struct pw_context *context, const char *section,
		int (*callback) (void *data, const char *location, const char *section,
			const char *str, size_t len),
		void *data);
/** emit callback for all matched properties. Since 0.3.46 */
int pw_context_conf_section_match_rules(struct pw_context *context, const char *section,
		const struct spa_dict *props,
		int (*callback) (void *data, const char *location, const char *action,
			const char *str, size_t len),
		void *data);

/** Get the context support objects */
const struct spa_support *pw_context_get_support(struct pw_context *context, uint32_t *n_support);

/** get the context main loop */
struct pw_loop *pw_context_get_main_loop(struct pw_context *context);

/** get the context data loop. Since 0.3.56 */
struct pw_data_loop *pw_context_get_data_loop(struct pw_context *context);

/** Get the work queue from the context: Since 0.3.26 */
struct pw_work_queue *pw_context_get_work_queue(struct pw_context *context);

/** Get the memmory pool from the context: Since 0.3.74 */
struct pw_mempool *pw_context_get_mempool(struct pw_context *context);

/** Iterate the globals of the context. The callback should return
 * 0 to fetch the next item, any other value stops the iteration and returns
 * the value. When all callbacks return 0, this function returns 0 when all
 * globals are iterated. */
int pw_context_for_each_global(struct pw_context *context,
			    int (*callback) (void *data, struct pw_global *global),
			    void *data);

/** Find a context global by id */
struct pw_global *pw_context_find_global(struct pw_context *context,	/**< the context */
				      uint32_t id		/**< the global id */);

/** add a spa library for the given factory_name regex */
int pw_context_add_spa_lib(struct pw_context *context, const char *factory_regex, const char *lib);

/** find the library name for a spa factory */
const char * pw_context_find_spa_lib(struct pw_context *context, const char *factory_name);

struct spa_handle *pw_context_load_spa_handle(struct pw_context *context,
		const char *factory_name,
		const struct spa_dict *info);


/** data for registering export functions */
struct pw_export_type {
	struct spa_list link;
	const char *type;
	struct pw_proxy * (*func) (struct pw_core *core,
		const char *type, const struct spa_dict *props, void *object,
		size_t user_data_size);
};

/** register a type that can be exported on a context_proxy. This is usually used by
 * extension modules */
int pw_context_register_export_type(struct pw_context *context, struct pw_export_type *type);
/** find information about registered export type */
const struct pw_export_type *pw_context_find_export_type(struct pw_context *context, const char *type);

/** add an object to the context */
int pw_context_set_object(struct pw_context *context, const char *type, void *value);
/** get an object from the context */
void *pw_context_get_object(struct pw_context *context, const char *type);

/**
 * \}
 */
#ifdef __cplusplus
}
#endif

#endif /* PIPEWIRE_CONTEXT_H */
