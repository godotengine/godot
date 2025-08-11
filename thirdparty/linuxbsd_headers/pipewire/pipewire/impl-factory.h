/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_IMPL_FACTORY_H
#define PIPEWIRE_IMPL_FACTORY_H

#ifdef __cplusplus
extern "C" {
#endif

/** \defgroup pw_impl_factory Factory Impl
 *
 * The factory is used to make objects on demand.
 */

/**
 * \addtogroup pw_impl_factory
 * \{
 */
struct pw_impl_factory;

#include <pipewire/context.h>
#include <pipewire/impl-client.h>
#include <pipewire/global.h>
#include <pipewire/properties.h>
#include <pipewire/resource.h>

/** Factory events, listen to them with \ref pw_impl_factory_add_listener */
struct pw_impl_factory_events {
#define PW_VERSION_IMPL_FACTORY_EVENTS	0
	uint32_t version;

	/** the factory is destroyed */
        void (*destroy) (void *data);
	/** the factory is freed */
        void (*free) (void *data);
	/** the factory is initialized */
        void (*initialized) (void *data);
};

struct pw_impl_factory_implementation {
#define PW_VERSION_IMPL_FACTORY_IMPLEMENTATION	0
	uint32_t version;

	/** The function to create an object from this factory */
	void *(*create_object) (void *data,
				struct pw_resource *resource,
				const char *type,
				uint32_t version,
				struct pw_properties *properties,
				uint32_t new_id);
};

struct pw_impl_factory *pw_context_create_factory(struct pw_context *context,
				  const char *name,
				  const char *type,
				  uint32_t version,
				  struct pw_properties *properties,
				  size_t user_data_size);

/** Get the factory properties */
const struct pw_properties *pw_impl_factory_get_properties(struct pw_impl_factory *factory);

/** Get the factory info */
const struct pw_factory_info *pw_impl_factory_get_info(struct pw_impl_factory *factory);

/** Update the factory properties */
int pw_impl_factory_update_properties(struct pw_impl_factory *factory, const struct spa_dict *dict);

int pw_impl_factory_register(struct pw_impl_factory *factory,
			struct pw_properties *properties);

void pw_impl_factory_destroy(struct pw_impl_factory *factory);

void *pw_impl_factory_get_user_data(struct pw_impl_factory *factory);

/** Get the global of this factory */
struct pw_global *pw_impl_factory_get_global(struct pw_impl_factory *factory);

/** Add an event listener */
void pw_impl_factory_add_listener(struct pw_impl_factory *factory,
			     struct spa_hook *listener,
			     const struct pw_impl_factory_events *events,
			     void *data);

void pw_impl_factory_set_implementation(struct pw_impl_factory *factory,
				   const struct pw_impl_factory_implementation *implementation,
				   void *data);

void *pw_impl_factory_create_object(struct pw_impl_factory *factory,
			       struct pw_resource *resource,
			       const char *type,
			       uint32_t version,
			       struct pw_properties *properties,
			       uint32_t new_id);

/** Find a factory by name */
struct pw_impl_factory *
pw_context_find_factory(struct pw_context *context	/**< the context */,
		     const char *name			/**< the factory name */);

/**
 * \}
 */

#ifdef __cplusplus
}
#endif

#endif /* PIPEWIRE_IMPL_FACTORY_H */
