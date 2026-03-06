/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_IMPL_PORT_H
#define PIPEWIRE_IMPL_PORT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/utils/hook.h>

/** \defgroup pw_impl_port Port Impl
 *
 * \brief A port can be used to link two nodes.
 */

/**
 * \addtogroup pw_impl_port
 * \{
 */
struct pw_impl_port;
struct pw_impl_link;
struct pw_control;

#include <pipewire/impl.h>

enum pw_impl_port_state {
	PW_IMPL_PORT_STATE_ERROR = -1,	/**< the port is in error */
	PW_IMPL_PORT_STATE_INIT = 0,		/**< the port is being created */
	PW_IMPL_PORT_STATE_CONFIGURE = 1,	/**< the port is ready for format negotiation */
	PW_IMPL_PORT_STATE_READY = 2,	/**< the port is ready for buffer allocation */
	PW_IMPL_PORT_STATE_PAUSED = 3,	/**< the port is paused */
};

/** Port events, use \ref pw_impl_port_add_listener */
struct pw_impl_port_events {
#define PW_VERSION_IMPL_PORT_EVENTS 3
	uint32_t version;

	/** The port is destroyed */
	void (*destroy) (void *data);

	/** The port is freed */
	void (*free) (void *data);

	/** The port is initialized */
	void (*initialized) (void *data);

	/** the port info changed */
	void (*info_changed) (void *data, const struct pw_port_info *info);

	/** a new link is added on this port */
	void (*link_added) (void *data, struct pw_impl_link *link);

	/** a link is removed from this port */
	void (*link_removed) (void *data, struct pw_impl_link *link);

	/** the state of the port changed */
	void (*state_changed) (void *data, enum pw_impl_port_state old,
			enum pw_impl_port_state state, const char *error);

	/** a control was added to the port */
	void (*control_added) (void *data, struct pw_control *control);

	/** a control was removed from the port */
	void (*control_removed) (void *data, struct pw_control *control);

	/** a parameter changed, since version 1 */
	void (*param_changed) (void *data, uint32_t id);

	/** latency changed. Since version 2 */
	void (*latency_changed) (void *data);
	/** tag changed. Since version 3 */
	void (*tag_changed) (void *data);
};

/** Create a new port
 * \return a newly allocated port */
struct pw_impl_port *
pw_context_create_port(struct pw_context *context,
	enum pw_direction direction,
	uint32_t port_id,
	const struct spa_port_info *info,
	size_t user_data_size);

/** Get the port direction */
enum pw_direction pw_impl_port_get_direction(struct pw_impl_port *port);

/** Get the port properties */
const struct pw_properties *pw_impl_port_get_properties(struct pw_impl_port *port);

/** Update the port properties */
int pw_impl_port_update_properties(struct pw_impl_port *port, const struct spa_dict *dict);

/** Get the port info */
const struct pw_port_info *pw_impl_port_get_info(struct pw_impl_port *port);

/** Get the port id */
uint32_t pw_impl_port_get_id(struct pw_impl_port *port);

/** Get the port state as a string */
const char *pw_impl_port_state_as_string(enum pw_impl_port_state state);

/** Get the port parent node or NULL when not yet set */
struct pw_impl_node *pw_impl_port_get_node(struct pw_impl_port *port);

/** check is a port has links, return 0 if not, 1 if it is linked */
int pw_impl_port_is_linked(struct pw_impl_port *port);

/** Add a port to a node */
int pw_impl_port_add(struct pw_impl_port *port, struct pw_impl_node *node);

/** Add an event listener on the port */
void pw_impl_port_add_listener(struct pw_impl_port *port,
			  struct spa_hook *listener,
			  const struct pw_impl_port_events *events,
			  void *data);

/**
 * \}
 */

#ifdef __cplusplus
}
#endif

#endif /* PIPEWIRE_IMPL_PORT_H */
