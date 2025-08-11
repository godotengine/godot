/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2016 Axis Communications <dev-gstreamer@axis.com> */
/*                         @author Linus Svensson <linus.svensson@axis.com> */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_IMPL_MODULE_H
#define PIPEWIRE_IMPL_MODULE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/utils/hook.h>

#include <pipewire/context.h>

#define PIPEWIRE_SYMBOL_MODULE_INIT "pipewire__module_init"
#define PIPEWIRE_MODULE_PREFIX "libpipewire-"

/** \defgroup pw_impl_module Module Impl
 *
 * A dynamically loadable module
 */

/**
 * \addtogroup pw_impl_module
 * \{
 */
struct pw_impl_module;

/** Module init function signature
 *
 * \param module A \ref pw_impl_module
 * \param args Arguments to the module
 * \return 0 on success, < 0 otherwise with an errno style error
 *
 * A module should provide an init function with this signature. This function
 * will be called when a module is loaded.
 */
typedef int (*pw_impl_module_init_func_t) (struct pw_impl_module *module, const char *args);

/** Module events added with \ref pw_impl_module_add_listener */
struct pw_impl_module_events {
#define PW_VERSION_IMPL_MODULE_EVENTS	0
	uint32_t version;

	/** The module is destroyed */
	void (*destroy) (void *data);
	/** The module is freed */
	void (*free) (void *data);
	/** The module is initialized */
	void (*initialized) (void *data);

	/** The module is registered. This is a good time to register
	 * objects created from the module. */
	void (*registered) (void *data);
};

struct pw_impl_module *
pw_context_load_module(struct pw_context *context,
	       const char *name,
	       const char *args,
	       struct pw_properties *properties);

/** Get the context of a module */
struct pw_context * pw_impl_module_get_context(struct pw_impl_module *module);

/** Get the global of a module */
struct pw_global * pw_impl_module_get_global(struct pw_impl_module *module);

/** Get the module properties */
const struct pw_properties *pw_impl_module_get_properties(struct pw_impl_module *module);

/** Update the module properties */
int pw_impl_module_update_properties(struct pw_impl_module *module, const struct spa_dict *dict);

/** Get the module info */
const struct pw_module_info *pw_impl_module_get_info(struct pw_impl_module *module);

/** Add an event listener to a module */
void pw_impl_module_add_listener(struct pw_impl_module *module,
			    struct spa_hook *listener,
			    const struct pw_impl_module_events *events,
			    void *data);

/** Destroy a module */
void pw_impl_module_destroy(struct pw_impl_module *module);

/** Schedule a destroy later on the main thread */
void pw_impl_module_schedule_destroy(struct pw_impl_module *module);

/**
 * \}
 */

#ifdef __cplusplus
}
#endif

#endif /* PIPEWIRE_IMPL_MODULE_H */
