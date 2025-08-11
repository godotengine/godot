/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_IMPL_LINK_H
#define PIPEWIRE_IMPL_LINK_H

#ifdef __cplusplus
extern "C" {
#endif

/** \defgroup pw_impl_link Link Impl
 *
 * \brief PipeWire link object.
 */

/**
 * \addtogroup pw_impl_link
 * \{
 */
struct pw_impl_link;
struct pw_impl_port;

#include <pipewire/impl.h>

/** link events added with \ref pw_impl_link_add_listener */
struct pw_impl_link_events {
#define PW_VERSION_IMPL_LINK_EVENTS	0
	uint32_t version;

	/** A link is destroyed */
	void (*destroy) (void *data);

	/** A link is freed */
	void (*free) (void *data);

	/** a Link is initialized */
	void (*initialized) (void *data);

	/** The info changed on a link */
	void (*info_changed) (void *data, const struct pw_link_info *info);

	/** The link state changed, \a error is only valid when the state is
	  * in error. */
	void (*state_changed) (void *data, enum pw_link_state old,
					   enum pw_link_state state, const char *error);

	/** A port is unlinked */
	void (*port_unlinked) (void *data, struct pw_impl_port *port);
};


/** Make a new link between two ports
 * \return a newly allocated link */
struct pw_impl_link *
pw_context_create_link(struct pw_context *context,		/**< the context object */
	    struct pw_impl_port *output,		/**< an output port */
	    struct pw_impl_port *input,		/**< an input port */
	    struct spa_pod *format_filter,	/**< an optional format filter */
	    struct pw_properties *properties	/**< extra properties */,
	    size_t user_data_size		/**< extra user data size */);

/** Destroy a link */
void pw_impl_link_destroy(struct pw_impl_link *link);

/** Add an event listener to \a link */
void pw_impl_link_add_listener(struct pw_impl_link *link,
			  struct spa_hook *listener,
			  const struct pw_impl_link_events *events,
			  void *data);

/** Finish link configuration and register */
int pw_impl_link_register(struct pw_impl_link *link,		/**< the link to register */
		     struct pw_properties *properties	/**< extra properties */);

/** Get the context of a link */
struct pw_context *pw_impl_link_get_context(struct pw_impl_link *link);

/** Get the user_data of a link, the size of the memory is given when
  * constructing the link */
void *pw_impl_link_get_user_data(struct pw_impl_link *link);

/** Get the link info */
const struct pw_link_info *pw_impl_link_get_info(struct pw_impl_link *link);

/** Get the global of the link */
struct pw_global *pw_impl_link_get_global(struct pw_impl_link *link);

/** Get the output port of the link */
struct pw_impl_port *pw_impl_link_get_output(struct pw_impl_link *link);

/** Get the input port of the link */
struct pw_impl_port *pw_impl_link_get_input(struct pw_impl_link *link);

/** Find the link between 2 ports */
struct pw_impl_link *pw_impl_link_find(struct pw_impl_port *output, struct pw_impl_port *input);

/**
 * \}
 */

#ifdef __cplusplus
}
#endif

#endif /* PIPEWIRE_IMPL_LINK_H */
