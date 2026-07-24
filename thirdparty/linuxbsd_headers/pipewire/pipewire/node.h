/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_NODE_H
#define PIPEWIRE_NODE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>
#include <errno.h>

#include <spa/utils/defs.h>
#include <spa/utils/hook.h>
#include <spa/node/command.h>
#include <spa/param/param.h>

#include <pipewire/proxy.h>

/** \defgroup pw_node Node
 * Node interface
 */

/**
 * \addtogroup pw_node
 * \{
 */
#define PW_TYPE_INTERFACE_Node	PW_TYPE_INFO_INTERFACE_BASE "Node"

#define PW_NODE_PERM_MASK	PW_PERM_RWXML

#define PW_VERSION_NODE		3
struct pw_node;

/** \enum pw_node_state The different node states */
enum pw_node_state {
	PW_NODE_STATE_ERROR = -1,	/**< error state */
	PW_NODE_STATE_CREATING = 0,	/**< the node is being created */
	PW_NODE_STATE_SUSPENDED = 1,	/**< the node is suspended, the device might
					 *   be closed */
	PW_NODE_STATE_IDLE = 2,		/**< the node is running but there is no active
					 *   port */
	PW_NODE_STATE_RUNNING = 3,	/**< the node is running */
};

/** Convert a \ref pw_node_state to a readable string */
const char * pw_node_state_as_string(enum pw_node_state state);

/** The node information. Extra information can be added in later versions */
struct pw_node_info {
	uint32_t id;				/**< id of the global */
	uint32_t max_input_ports;		/**< maximum number of inputs */
	uint32_t max_output_ports;		/**< maximum number of outputs */
#define PW_NODE_CHANGE_MASK_INPUT_PORTS		(1 << 0)
#define PW_NODE_CHANGE_MASK_OUTPUT_PORTS	(1 << 1)
#define PW_NODE_CHANGE_MASK_STATE		(1 << 2)
#define PW_NODE_CHANGE_MASK_PROPS		(1 << 3)
#define PW_NODE_CHANGE_MASK_PARAMS		(1 << 4)
#define PW_NODE_CHANGE_MASK_ALL			((1 << 5)-1)
	uint64_t change_mask;			/**< bitfield of changed fields since last call */
	uint32_t n_input_ports;			/**< number of inputs */
	uint32_t n_output_ports;		/**< number of outputs */
	enum pw_node_state state;		/**< the current state of the node */
	const char *error;			/**< an error reason if \a state is error */
	struct spa_dict *props;			/**< the properties of the node */
	struct spa_param_info *params;		/**< parameters */
	uint32_t n_params;			/**< number of items in \a params */
};

struct pw_node_info *
pw_node_info_update(struct pw_node_info *info,
		const struct pw_node_info *update);

struct pw_node_info *
pw_node_info_merge(struct pw_node_info *info,
		const struct pw_node_info *update, bool reset);

void
pw_node_info_free(struct pw_node_info *info);

#define PW_NODE_EVENT_INFO	0
#define PW_NODE_EVENT_PARAM	1
#define PW_NODE_EVENT_NUM	2

/** Node events */
struct pw_node_events {
#define PW_VERSION_NODE_EVENTS	0
	uint32_t version;
	/**
	 * Notify node info
	 *
	 * \param info info about the node
	 */
	void (*info) (void *data, const struct pw_node_info *info);
	/**
	 * Notify a node param
	 *
	 * Event emitted as a result of the enum_params method.
	 *
	 * \param seq the sequence number of the request
	 * \param id the param id
	 * \param index the param index
	 * \param next the param index of the next param
	 * \param param the parameter
	 */
	void (*param) (void *data, int seq,
		      uint32_t id, uint32_t index, uint32_t next,
		      const struct spa_pod *param);
};

#define PW_NODE_METHOD_ADD_LISTENER	0
#define PW_NODE_METHOD_SUBSCRIBE_PARAMS	1
#define PW_NODE_METHOD_ENUM_PARAMS	2
#define PW_NODE_METHOD_SET_PARAM	3
#define PW_NODE_METHOD_SEND_COMMAND	4
#define PW_NODE_METHOD_NUM		5

/** Node methods */
struct pw_node_methods {
#define PW_VERSION_NODE_METHODS		0
	uint32_t version;

	int (*add_listener) (void *object,
			struct spa_hook *listener,
			const struct pw_node_events *events,
			void *data);
	/**
	 * Subscribe to parameter changes
	 *
	 * Automatically emit param events for the given ids when
	 * they are changed.
	 *
	 * \param ids an array of param ids
	 * \param n_ids the number of ids in \a ids
	 *
	 * This requires X permissions on the node.
	 */
	int (*subscribe_params) (void *object, uint32_t *ids, uint32_t n_ids);

	/**
	 * Enumerate node parameters
	 *
	 * Start enumeration of node parameters. For each param, a
	 * param event will be emitted.
	 *
	 * \param seq a sequence number to place in the reply
	 * \param id the parameter id to enum or PW_ID_ANY for all
	 * \param start the start index or 0 for the first param
	 * \param num the maximum number of params to retrieve
	 * \param filter a param filter or NULL
	 *
	 * This requires X permissions on the node.
	 */
	int (*enum_params) (void *object, int seq, uint32_t id,
			uint32_t start, uint32_t num,
			const struct spa_pod *filter);

	/**
	 * Set a parameter on the node
	 *
	 * \param id the parameter id to set
	 * \param flags extra parameter flags
	 * \param param the parameter to set
	 *
	 * This requires X and W permissions on the node.
	 */
	int (*set_param) (void *object, uint32_t id, uint32_t flags,
			const struct spa_pod *param);

	/**
	 * Send a command to the node
	 *
	 * \param command the command to send
	 *
	 * This requires X and W permissions on the node.
	 */
	int (*send_command) (void *object, const struct spa_command *command);
};

#define pw_node_method(o,method,version,...)				\
({									\
	int _res = -ENOTSUP;						\
	spa_interface_call_res((struct spa_interface*)o,		\
			struct pw_node_methods, _res,			\
			method, version, ##__VA_ARGS__);		\
	_res;								\
})

/** Node */
#define pw_node_add_listener(c,...)	pw_node_method(c,add_listener,0,__VA_ARGS__)
#define pw_node_subscribe_params(c,...)	pw_node_method(c,subscribe_params,0,__VA_ARGS__)
#define pw_node_enum_params(c,...)	pw_node_method(c,enum_params,0,__VA_ARGS__)
#define pw_node_set_param(c,...)	pw_node_method(c,set_param,0,__VA_ARGS__)
#define pw_node_send_command(c,...)	pw_node_method(c,send_command,0,__VA_ARGS__)

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* PIPEWIRE_NODE_H */
