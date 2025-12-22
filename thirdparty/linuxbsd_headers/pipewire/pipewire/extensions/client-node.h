/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_EXT_CLIENT_NODE_H
#define PIPEWIRE_EXT_CLIENT_NODE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <spa/utils/defs.h>
#include <spa/param/param.h>

/** \defgroup pw_client_node Client Node
 * Client node interface
 */

/**
 * \addtogroup pw_client_node
 * \{
 */
#define PW_TYPE_INTERFACE_ClientNode		PW_TYPE_INFO_INTERFACE_BASE "ClientNode"

#define PW_VERSION_CLIENT_NODE			5
struct pw_client_node;

#define PW_EXTENSION_MODULE_CLIENT_NODE		PIPEWIRE_MODULE_PREFIX "module-client-node"

/** information about a buffer */
struct pw_client_node_buffer {
	uint32_t mem_id;		/**< the memory id for the metadata */
	uint32_t offset;		/**< offset in memory */
	uint32_t size;			/**< size in memory */
	struct spa_buffer *buffer;	/**< buffer describing metadata and buffer memory */
};

#define PW_CLIENT_NODE_EVENT_TRANSPORT		0
#define PW_CLIENT_NODE_EVENT_SET_PARAM		1
#define PW_CLIENT_NODE_EVENT_SET_IO		2
#define PW_CLIENT_NODE_EVENT_EVENT		3
#define PW_CLIENT_NODE_EVENT_COMMAND		4
#define PW_CLIENT_NODE_EVENT_ADD_PORT		5
#define PW_CLIENT_NODE_EVENT_REMOVE_PORT	6
#define PW_CLIENT_NODE_EVENT_PORT_SET_PARAM	7
#define PW_CLIENT_NODE_EVENT_PORT_USE_BUFFERS	8
#define PW_CLIENT_NODE_EVENT_PORT_SET_IO	9
#define PW_CLIENT_NODE_EVENT_SET_ACTIVATION	10
#define PW_CLIENT_NODE_EVENT_PORT_SET_MIX_INFO	11
#define PW_CLIENT_NODE_EVENT_NUM		12

/** \ref pw_client_node events */
struct pw_client_node_events {
#define PW_VERSION_CLIENT_NODE_EVENTS		1
	uint32_t version;
	/**
	 * Notify of a new transport area
	 *
	 * The transport area is used to signal the client and the server.
	 *
	 * \param readfd fd for signal data can be read
	 * \param writefd fd for signal data can be written
	 * \param mem_id id for activation memory
	 * \param offset offset of activation memory
	 * \param size size of activation memory
	 */
	int (*transport) (void *data,
			  int readfd,
			  int writefd,
			  uint32_t mem_id,
			  uint32_t offset,
			  uint32_t size);
	/**
	 * Notify of a property change
	 *
	 * When the server configures the properties on the node
	 * this event is sent
	 *
	 * \param id the id of the parameter
	 * \param flags parameter flags
	 * \param param the param to set
	 */
	int (*set_param) (void *data,
			  uint32_t id, uint32_t flags,
			  const struct spa_pod *param);
	/**
	 * Configure an IO area for the client
	 *
	 * IO areas are identified with an id and are used to
	 * exchange state between client and server
	 *
	 * \param id the id of the io area
	 * \param mem_id the id of the memory to use
	 * \param offset offset of io area in memory
	 * \param size size of the io area
	 */
	int (*set_io) (void *data,
			uint32_t id,
			uint32_t mem_id,
			uint32_t offset,
			uint32_t size);
	/**
	 * Receive an event from the client node
	 * \param event the received event */
	int (*event) (void *data, const struct spa_event *event);
	/**
	 * Notify of a new node command
	 *
	 * \param command the command
	 */
	int (*command) (void *data, const struct spa_command *command);
	/**
	 * A new port was added to the node
	 *
	 * The server can at any time add a port to the node when there
	 * are free ports available.
	 *
	 * \param direction the direction of the port
	 * \param port_id the new port id
	 * \param props extra properties
	 */
	int (*add_port) (void *data,
			  enum spa_direction direction,
			  uint32_t port_id,
			  const struct spa_dict *props);
	/**
	 * A port was removed from the node
	 *
	 * \param direction a port direction
	 * \param port_id the remove port id
	 */
	int (*remove_port) (void *data,
			     enum spa_direction direction,
			     uint32_t port_id);
	/**
	 * A parameter was configured on the port
	 *
	 * \param direction a port direction
	 * \param port_id the port id
	 * \param id the id of the parameter
	 * \param flags flags used when setting the param
	 * \param param the new param
	 */
	int (*port_set_param) (void *data,
				enum spa_direction direction,
				uint32_t port_id,
				uint32_t id, uint32_t flags,
				const struct spa_pod *param);
	/**
	 * Notify the port of buffers
	 *
	 * \param direction a port direction
	 * \param port_id the port id
	 * \param mix_id the mixer port id
	 * \param n_buffer the number of buffers
	 * \param buffers and array of buffer descriptions
	 */
	int (*port_use_buffers) (void *data,
				  enum spa_direction direction,
				  uint32_t port_id,
				  uint32_t mix_id,
				  uint32_t flags,
				  uint32_t n_buffers,
				  struct pw_client_node_buffer *buffers);
	/**
	 * Configure the io area with \a id of \a port_id.
	 *
	 * \param direction the direction of the port
	 * \param port_id the port id
	 * \param mix_id the mixer port id
	 * \param id the id of the io area to set
	 * \param mem_id the id of the memory to use
	 * \param offset offset of io area in memory
	 * \param size size of the io area
	 */
	int (*port_set_io) (void *data,
			     enum spa_direction direction,
			     uint32_t port_id,
			     uint32_t mix_id,
			     uint32_t id,
			     uint32_t mem_id,
			     uint32_t offset,
			     uint32_t size);

	/**
	 * Notify the activation record of the next
	 * node to trigger
	 *
	 * \param node_id the peer node id
	 * \param signalfd the fd to wake up the peer
	 * \param mem_id the mem id of the memory
	 * \param the offset in \a mem_id to map
	 * \param the size of \a mem_id to map
	 */
	int (*set_activation) (void *data,
				uint32_t node_id,
				int signalfd,
				uint32_t mem_id,
				uint32_t offset,
				uint32_t size);

	/**
	 * Notify about the peer of mix_id
	 *
	 * \param direction the direction of the port
	 * \param port_id the port id
	 * \param mix_id the mix id
	 * \param peer_id the id of the peer port
	 * \param props extra properties
	 *
	 * Since version 4:1
	 */
	int (*port_set_mix_info) (void *data,
			enum spa_direction direction,
			uint32_t port_id,
			uint32_t mix_id,
			uint32_t peer_id,
			const struct spa_dict *props);
};

#define PW_CLIENT_NODE_METHOD_ADD_LISTENER	0
#define PW_CLIENT_NODE_METHOD_GET_NODE		1
#define PW_CLIENT_NODE_METHOD_UPDATE		2
#define PW_CLIENT_NODE_METHOD_PORT_UPDATE	3
#define PW_CLIENT_NODE_METHOD_SET_ACTIVE	4
#define PW_CLIENT_NODE_METHOD_EVENT		5
#define PW_CLIENT_NODE_METHOD_PORT_BUFFERS	6
#define PW_CLIENT_NODE_METHOD_NUM		7

/** \ref pw_client_node methods */
struct pw_client_node_methods {
#define PW_VERSION_CLIENT_NODE_METHODS		0
	uint32_t version;

	int (*add_listener) (void *object,
			struct spa_hook *listener,
			const struct pw_client_node_events *events,
			void *data);
	/** get the node object
	 */
	struct pw_node * (*get_node) (void *object, uint32_t version, size_t user_data_size);
	/**
	 * Update the node ports and properties
	 *
	 * Update the maximum number of ports and the params of the
	 * client node.
	 * \param change_mask bitfield with changed parameters
	 * \param max_input_ports new max input ports
	 * \param max_output_ports new max output ports
	 * \param params new params
	 */
	int (*update) (void *object,
#define PW_CLIENT_NODE_UPDATE_PARAMS		(1 << 0)
#define PW_CLIENT_NODE_UPDATE_INFO		(1 << 1)
			uint32_t change_mask,
			uint32_t n_params,
			const struct spa_pod **params,
			const struct spa_node_info *info);

	/**
	 * Update a node port
	 *
	 * Update the information of one port of a node.
	 * \param direction the direction of the port
	 * \param port_id the port id to update
	 * \param change_mask a bitfield of changed items
	 * \param n_params number of port parameters
	 * \param params array of port parameters
	 * \param info port information
	 */
	int (*port_update) (void *object,
			     enum spa_direction direction,
			     uint32_t port_id,
#define PW_CLIENT_NODE_PORT_UPDATE_PARAMS            (1 << 0)
#define PW_CLIENT_NODE_PORT_UPDATE_INFO              (1 << 1)
			     uint32_t change_mask,
			     uint32_t n_params,
			     const struct spa_pod **params,
			     const struct spa_port_info *info);
	/**
	 * Activate or deactivate the node
	 */
	int (*set_active) (void *object, bool active);
	/**
	 * Send an event to the node
	 * \param event the event to send
	 */
	int (*event) (void *object, const struct spa_event *event);

	/**
	 * Send allocated buffers
	 */
	int (*port_buffers) (void *object,
			  enum spa_direction direction,
			  uint32_t port_id,
			  uint32_t mix_id,
			  uint32_t n_buffers,
			  struct spa_buffer **buffers);
};


#define pw_client_node_method(o,method,version,...)			\
({									\
	int _res = -ENOTSUP;						\
	spa_interface_call_res((struct spa_interface*)o,		\
			struct pw_client_node_methods, _res,		\
			method, version, ##__VA_ARGS__);		\
	_res;								\
})

#define pw_client_node_add_listener(c,...)	pw_client_node_method(c,add_listener,0,__VA_ARGS__)

static inline struct pw_node *
pw_client_node_get_node(struct pw_client_node *p, uint32_t version, size_t user_data_size)
{
	struct pw_node *res = NULL;
	spa_interface_call_res((struct spa_interface*)p,
			struct pw_client_node_methods, res,
			get_node, 0, version, user_data_size);
	return res;
}

#define pw_client_node_update(c,...)		pw_client_node_method(c,update,0,__VA_ARGS__)
#define pw_client_node_port_update(c,...)	pw_client_node_method(c,port_update,0,__VA_ARGS__)
#define pw_client_node_set_active(c,...)	pw_client_node_method(c,set_active,0,__VA_ARGS__)
#define pw_client_node_event(c,...)		pw_client_node_method(c,event,0,__VA_ARGS__)
#define pw_client_node_port_buffers(c,...)	pw_client_node_method(c,port_buffers,0,__VA_ARGS__)

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* PIPEWIRE_EXT_CLIENT_NODE_H */
