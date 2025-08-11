/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_IMPL_NODE_H
#define PIPEWIRE_IMPL_NODE_H

#ifdef __cplusplus
extern "C" {
#endif

/** \defgroup pw_impl_node Node Impl
 *
 * The node object processes data. The node has a list of
 * input and output ports (\ref pw_impl_port) on which it
 * will receive and send out buffers respectively.
 */
/**
 * \addtogroup pw_impl_node
 * \{
 */
struct pw_impl_node;
struct pw_impl_port;

#include <spa/node/node.h>
#include <spa/node/event.h>

#include <pipewire/impl.h>

/** Node events, listen to them with \ref pw_impl_node_add_listener */
struct pw_impl_node_events {
#define PW_VERSION_IMPL_NODE_EVENTS	0
	uint32_t version;

	/** the node is destroyed */
        void (*destroy) (void *data);
	/** the node is about to be freed */
        void (*free) (void *data);
	/** the node is initialized */
        void (*initialized) (void *data);

	/** a port is being initialized on the node */
        void (*port_init) (void *data, struct pw_impl_port *port);
	/** a port was added */
        void (*port_added) (void *data, struct pw_impl_port *port);
	/** a port was removed */
        void (*port_removed) (void *data, struct pw_impl_port *port);

	/** the node info changed */
	void (*info_changed) (void *data, const struct pw_node_info *info);
	/** a port on the node changed info */
	void (*port_info_changed) (void *data, struct pw_impl_port *port,
			const struct pw_port_info *info);
	/** the node active state changed */
	void (*active_changed) (void *data, bool active);

	/** a new state is requested on the node */
	void (*state_request) (void *data, enum pw_node_state state);
	/** the state of the node changed */
	void (*state_changed) (void *data, enum pw_node_state old,
			       enum pw_node_state state, const char *error);

        /** a result was received */
	void (*result) (void *data, int seq, int res, uint32_t type, const void *result);

        /** an event is emitted */
	void (*event) (void *data, const struct spa_event *event);

	/** the driver of the node changed */
	void (*driver_changed) (void *data, struct pw_impl_node *old, struct pw_impl_node *driver);

	/** a peer was added */
	void (*peer_added) (void *data, struct pw_impl_node *peer);
	/** a peer was removed */
	void (*peer_removed) (void *data, struct pw_impl_node *peer);
};

struct pw_impl_node_rt_events {
#define PW_VERSION_IMPL_NODE_RT_EVENTS	0
	uint32_t version;
	/** the node is drained */
	void (*drained) (void *data);
	/** the node had an xrun */
	void (*xrun) (void *data);
	/** the driver node starts processing */
	void (*start) (void *data);
	/** the driver node completed processing */
	void (*complete) (void *data);
	/** the driver node did not complete processing */
	void (*incomplete) (void *data);
	/** the node had */
	void (*timeout) (void *data);
};

/** Create a new node */
struct pw_impl_node *
pw_context_create_node(struct pw_context *context,	/**< the context */
	    struct pw_properties *properties,	/**< extra properties */
	    size_t user_data_size		/**< user data size */);

/** Complete initialization of the node and register */
int pw_impl_node_register(struct pw_impl_node *node,		/**< node to register */
		     struct pw_properties *properties	/**< extra properties */);

/** Destroy a node */
void pw_impl_node_destroy(struct pw_impl_node *node);

/** Get the node info */
const struct pw_node_info *pw_impl_node_get_info(struct pw_impl_node *node);

/** Get node user_data. The size of the memory was given in \ref pw_context_create_node */
void * pw_impl_node_get_user_data(struct pw_impl_node *node);

/** Get the context of this node */
struct pw_context *pw_impl_node_get_context(struct pw_impl_node *node);

/** Get the global of this node */
struct pw_global *pw_impl_node_get_global(struct pw_impl_node *node);

/** Get the node properties */
const struct pw_properties *pw_impl_node_get_properties(struct pw_impl_node *node);

/** Update the node properties */
int pw_impl_node_update_properties(struct pw_impl_node *node, const struct spa_dict *dict);

/** Set the node implementation */
int pw_impl_node_set_implementation(struct pw_impl_node *node, struct spa_node *spa_node);

/** Get the node implementation */
struct spa_node *pw_impl_node_get_implementation(struct pw_impl_node *node);

/** Add an event listener */
void pw_impl_node_add_listener(struct pw_impl_node *node,
			  struct spa_hook *listener,
			  const struct pw_impl_node_events *events,
			  void *data);

/** Add an rt_event listener */
void pw_impl_node_add_rt_listener(struct pw_impl_node *node,
			  struct spa_hook *listener,
			  const struct pw_impl_node_rt_events *events,
			  void *data);
void pw_impl_node_remove_rt_listener(struct pw_impl_node *node,
			  struct spa_hook *listener);

/** Iterate the ports in the given direction. The callback should return
 * 0 to fetch the next item, any other value stops the iteration and returns
 * the value. When all callbacks return 0, this function returns 0 when all
 * items are iterated. */
int pw_impl_node_for_each_port(struct pw_impl_node *node,
			  enum pw_direction direction,
			  int (*callback) (void *data, struct pw_impl_port *port),
			  void *data);

int pw_impl_node_for_each_param(struct pw_impl_node *node,
			   int seq, uint32_t param_id,
			   uint32_t index, uint32_t max,
			   const struct spa_pod *filter,
			   int (*callback) (void *data, int seq,
					    uint32_t id, uint32_t index, uint32_t next,
					    struct spa_pod *param),
			   void *data);

/** Find the port with direction and port_id or NULL when not found. Passing
 * PW_ID_ANY for port_id will return any port, preferably an unlinked one. */
struct pw_impl_port *
pw_impl_node_find_port(struct pw_impl_node *node, enum pw_direction direction, uint32_t port_id);

/** Get a free unused port_id from the node */
uint32_t pw_impl_node_get_free_port_id(struct pw_impl_node *node, enum pw_direction direction);

int pw_impl_node_initialized(struct pw_impl_node *node);

/** Set a node active. This will start negotiation with all linked active
  * nodes and start data transport */
int pw_impl_node_set_active(struct pw_impl_node *node, bool active);

/** Check if a node is active */
bool pw_impl_node_is_active(struct pw_impl_node *node);

/** Check if a node is active, Since 0.3.39 */
int pw_impl_node_send_command(struct pw_impl_node *node, const struct spa_command *command);

/** Set a param on the node, Since 0.3.65 */
int pw_impl_node_set_param(struct pw_impl_node *node,
		uint32_t id, uint32_t flags, const struct spa_pod *param);
/**
 * \}
 */

#ifdef __cplusplus
}
#endif

#endif /* PIPEWIRE_IMPL_NODE_H */
