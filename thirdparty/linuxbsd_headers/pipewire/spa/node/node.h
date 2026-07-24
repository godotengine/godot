/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_NODE_H
#define SPA_NODE_H

#ifdef __cplusplus
extern "C" {
#endif

/** \defgroup spa_node Node
 *
 * A spa_node is a component that can consume and produce buffers.
 */

/**
 * \addtogroup spa_node
 * \{
 */

#include <errno.h>
#include <spa/utils/defs.h>
#include <spa/utils/type.h>
#include <spa/utils/hook.h>
#include <spa/buffer/buffer.h>
#include <spa/node/event.h>
#include <spa/node/command.h>


#define SPA_TYPE_INTERFACE_Node		SPA_TYPE_INFO_INTERFACE_BASE "Node"

#define SPA_VERSION_NODE		0
struct spa_node { struct spa_interface iface; };

/**
 * Node information structure
 *
 * Contains the basic node information.
 */
struct spa_node_info {
	uint32_t max_input_ports;
	uint32_t max_output_ports;
#define SPA_NODE_CHANGE_MASK_FLAGS		(1u<<0)
#define SPA_NODE_CHANGE_MASK_PROPS		(1u<<1)
#define SPA_NODE_CHANGE_MASK_PARAMS		(1u<<2)
	uint64_t change_mask;

#define SPA_NODE_FLAG_RT			(1u<<0)	/**< node can do real-time processing */
#define SPA_NODE_FLAG_IN_DYNAMIC_PORTS		(1u<<1)	/**< input ports can be added/removed */
#define SPA_NODE_FLAG_OUT_DYNAMIC_PORTS		(1u<<2)	/**< output ports can be added/removed */
#define SPA_NODE_FLAG_IN_PORT_CONFIG		(1u<<3)	/**< input ports can be reconfigured with
							  *  PortConfig parameter */
#define SPA_NODE_FLAG_OUT_PORT_CONFIG		(1u<<4)	/**< output ports can be reconfigured with
							  *  PortConfig parameter */
#define SPA_NODE_FLAG_NEED_CONFIGURE		(1u<<5)	/**< node needs configuration before it can
							  *  be started. */
#define SPA_NODE_FLAG_ASYNC			(1u<<6)	/**< the process function might not
							  *  immediately produce or consume data
							  *  but might offload the work to a worker
							  *  thread. */
	uint64_t flags;
	struct spa_dict *props;			/**< extra node properties */
	struct spa_param_info *params;		/**< parameter information */
	uint32_t n_params;			/**< number of items in \a params */
};

#define SPA_NODE_INFO_INIT()	((struct spa_node_info) { 0, })

/**
 * Port information structure
 *
 * Contains the basic port information.
 */
struct spa_port_info {
#define SPA_PORT_CHANGE_MASK_FLAGS		(1u<<0)
#define SPA_PORT_CHANGE_MASK_RATE		(1u<<1)
#define SPA_PORT_CHANGE_MASK_PROPS		(1u<<2)
#define SPA_PORT_CHANGE_MASK_PARAMS		(1u<<3)
	uint64_t change_mask;

#define SPA_PORT_FLAG_REMOVABLE			(1u<<0)	/**< port can be removed */
#define SPA_PORT_FLAG_OPTIONAL			(1u<<1)	/**< processing on port is optional */
#define SPA_PORT_FLAG_CAN_ALLOC_BUFFERS		(1u<<2)	/**< the port can allocate buffer data */
#define SPA_PORT_FLAG_IN_PLACE			(1u<<3)	/**< the port can process data in-place and
							 *   will need a writable input buffer */
#define SPA_PORT_FLAG_NO_REF			(1u<<4)	/**< the port does not keep a ref on the buffer.
							 *   This means the node will always completely
							 *   consume the input buffer and it will be
							 *   recycled after process. */
#define SPA_PORT_FLAG_LIVE			(1u<<5)	/**< output buffers from this port are
							 *   timestamped against a live clock. */
#define SPA_PORT_FLAG_PHYSICAL			(1u<<6)	/**< connects to some device */
#define SPA_PORT_FLAG_TERMINAL			(1u<<7)	/**< data was not created from this port
							 *   or will not be made available on another
							 *   port */
#define SPA_PORT_FLAG_DYNAMIC_DATA		(1u<<8)	/**< data pointer on buffers can be changed.
							 *   Only the buffer data marked as DYNAMIC
							 *   can be changed. */
	uint64_t flags;				/**< port flags */
	struct spa_fraction rate;		/**< rate of sequence numbers on port */
	const struct spa_dict *props;		/**< extra port properties */
	struct spa_param_info *params;		/**< parameter information */
	uint32_t n_params;			/**< number of items in \a params */
};

#define SPA_PORT_INFO_INIT()	((struct spa_port_info) { 0, })

#define SPA_RESULT_TYPE_NODE_ERROR	1
#define SPA_RESULT_TYPE_NODE_PARAMS	2

/** an error result */
struct spa_result_node_error {
	const char *message;
};

/** the result of enum_params or port_enum_params. */
struct spa_result_node_params {
	uint32_t id;		/**< id of parameter */
	uint32_t index;		/**< index of parameter */
	uint32_t next;		/**< next index of iteration */
	struct spa_pod *param;	/**< the result param */
};

#define SPA_NODE_EVENT_INFO		0
#define SPA_NODE_EVENT_PORT_INFO	1
#define SPA_NODE_EVENT_RESULT		2
#define SPA_NODE_EVENT_EVENT		3
#define SPA_NODE_EVENT_NUM		4

/** events from the spa_node.
 *
 * All event are called from the main thread and multiple
 * listeners can be registered for the events with
 * spa_node_add_listener().
 */
struct spa_node_events {
#define SPA_VERSION_NODE_EVENTS	0
	uint32_t version;	/**< version of this structure */

	/** Emitted when info changes */
	void (*info) (void *data, const struct spa_node_info *info);

	/** Emitted when port info changes, NULL when port is removed */
	void (*port_info) (void *data,
			enum spa_direction direction, uint32_t port,
			const struct spa_port_info *info);

	/** notify a result.
	 *
	 * Some methods will trigger a result event with an optional
	 * result of the given type. Look at the documentation of the
	 * method to know when to expect a result event.
	 *
	 * The result event can be called synchronously, as an event
	 * called from inside the method itself, in which case the seq
	 * number passed to the method will be passed unchanged.
	 *
	 * The result event will be called asynchronously when the
	 * method returned an async return value. In this case, the seq
	 * number in the result will match the async return value of
	 * the method call. Users should match the seq number from
	 * request to the reply.
	 */
	void (*result) (void *data, int seq, int res,
			uint32_t type, const void *result);

	/**
	 * \param node a spa_node
	 * \param event the event that was emitted
	 *
	 * This will be called when an out-of-bound event is notified
	 * on \a node.
	 */
	void (*event) (void *data, const struct spa_event *event);
};

#define SPA_NODE_CALLBACK_READY		0
#define SPA_NODE_CALLBACK_REUSE_BUFFER	1
#define SPA_NODE_CALLBACK_XRUN		2
#define SPA_NODE_CALLBACK_NUM		3

/** Node callbacks
 *
 * Callbacks are called from the real-time data thread. Only
 * one callback structure can be set on an spa_node.
 */
struct spa_node_callbacks {
#define SPA_VERSION_NODE_CALLBACKS	0
	uint32_t version;
	/**
	 * \param node a spa_node
	 *
	 * The node is ready for processing.
	 *
	 * When this function is NULL, synchronous operation is requested
	 * on the ports.
	 */
	int (*ready) (void *data, int state);

	/**
	 * \param node a spa_node
	 * \param port_id an input port_id
	 * \param buffer_id the buffer id to be reused
	 *
	 * The node has a buffer that can be reused.
	 *
	 * When this function is NULL, the buffers to reuse will be set in
	 * the io area of the input ports.
	 */
	int (*reuse_buffer) (void *data,
			     uint32_t port_id,
			     uint32_t buffer_id);

	/**
	 * \param data user data
	 * \param trigger the timestamp in microseconds when the xrun happened
	 * \param delay the amount of microseconds of xrun.
	 * \param info an object with extra info (NULL for now)
	 *
	 * The node has encountered an over or underrun
	 *
	 * The info contains an object with more information
	 */
	int (*xrun) (void *data, uint64_t trigger, uint64_t delay,
			struct spa_pod *info);
};


/** flags that can be passed to set_param and port_set_param functions */
#define SPA_NODE_PARAM_FLAG_TEST_ONLY	(1 << 0)	/**< Just check if the param is accepted */
#define SPA_NODE_PARAM_FLAG_FIXATE	(1 << 1)	/**< Fixate the non-optional unset fields */
#define SPA_NODE_PARAM_FLAG_NEAREST	(1 << 2)	/**< Allow set fields to be rounded to the
							  *  nearest allowed field value. */

/** flags to pass to the use_buffers functions */
#define SPA_NODE_BUFFERS_FLAG_ALLOC	(1 << 0)	/**< Allocate memory for the buffers. This flag
							  *  is ignored when the port does not have the
							  *  SPA_PORT_FLAG_CAN_ALLOC_BUFFERS set. */


#define SPA_NODE_METHOD_ADD_LISTENER		0
#define SPA_NODE_METHOD_SET_CALLBACKS		1
#define SPA_NODE_METHOD_SYNC			2
#define SPA_NODE_METHOD_ENUM_PARAMS		3
#define SPA_NODE_METHOD_SET_PARAM		4
#define SPA_NODE_METHOD_SET_IO			5
#define SPA_NODE_METHOD_SEND_COMMAND		6
#define SPA_NODE_METHOD_ADD_PORT		7
#define SPA_NODE_METHOD_REMOVE_PORT		8
#define SPA_NODE_METHOD_PORT_ENUM_PARAMS	9
#define SPA_NODE_METHOD_PORT_SET_PARAM		10
#define SPA_NODE_METHOD_PORT_USE_BUFFERS	11
#define SPA_NODE_METHOD_PORT_SET_IO		12
#define SPA_NODE_METHOD_PORT_REUSE_BUFFER	13
#define SPA_NODE_METHOD_PROCESS			14
#define SPA_NODE_METHOD_NUM			15

/**
 * Node methods
 */
struct spa_node_methods {
	/* the version of the node methods. This can be used to expand this
	 * structure in the future */
#define SPA_VERSION_NODE_METHODS	0
	uint32_t version;

	/**
	 * Adds an event listener on \a node.
	 *
	 * Setting the events will trigger the info event and a
	 * port_info event for each managed port on the new
	 * listener.
	 *
	 * \param node a #spa_node
	 * \param listener a listener
	 * \param events a struct \ref spa_node_events
	 * \param data data passed as first argument in functions of \a events
	 * \return 0 on success
	 *	   < 0 errno on error
	 */
	int (*add_listener) (void *object,
			struct spa_hook *listener,
			const struct spa_node_events *events,
			void *data);
	/**
	 * Set callbacks to on \a node.
	 * if \a callbacks is NULL, the current callbacks are removed.
	 *
	 * This function must be called from the main thread.
	 *
	 * All callbacks are called from the data thread.
	 *
	 * \param node a spa_node
	 * \param callbacks callbacks to set
	 * \return 0 on success
	 *         -EINVAL when node is NULL
	 */
	int (*set_callbacks) (void *object,
			      const struct spa_node_callbacks *callbacks,
			      void *data);
	/**
	 * Perform a sync operation.
	 *
	 * This method will emit the result event with the given sequence
	 * number synchronously or with the returned async return value
	 * asynchronously.
	 *
	 * Because all methods are serialized in the node, this can be used
	 * to wait for completion of all previous method calls.
	 *
	 * \param seq a sequence number
	 * \return 0 on success
	 *         -EINVAL when node is NULL
	 *         an async result
	 */
	int (*sync) (void *object, int seq);

	/**
	 * Enumerate the parameters of a node.
	 *
	 * Parameters are identified with an \a id. Some parameters can have
	 * multiple values, see the documentation of the parameter id.
	 *
	 * Parameters can be filtered by passing a non-NULL \a filter.
	 *
	 * The function will emit the result event up to \a max times with
	 * the result value. The seq in the result will either be the \a seq
	 * number when executed synchronously or the async return value of
	 * this function when executed asynchronously.
	 *
	 * This function must be called from the main thread.
	 *
	 * \param node a \ref spa_node
	 * \param seq a sequence number to pass to the result event when
	 *	this method is executed synchronously.
	 * \param id the param id to enumerate
	 * \param start the index of enumeration, pass 0 for the first item
	 * \param max the maximum number of parameters to enumerate
	 * \param filter and optional filter to use
	 *
	 * \return 0 when no more items can be iterated.
	 *         -EINVAL when invalid arguments are given
	 *         -ENOENT the parameter \a id is unknown
	 *         -ENOTSUP when there are no parameters
	 *                 implemented on \a node
	 *         an async return value when the result event will be
	 *             emitted later.
	 */
	int (*enum_params) (void *object, int seq,
			    uint32_t id, uint32_t start, uint32_t max,
			    const struct spa_pod *filter);

	/**
	 * Set the configurable parameter in \a node.
	 *
	 * Usually, \a param will be obtained from enum_params and then
	 * modified but it is also possible to set another spa_pod
	 * as long as its keys and types match a supported object.
	 *
	 * Objects with property keys that are not known are ignored.
	 *
	 * This function must be called from the main thread.
	 *
	 * \param node a \ref spa_node
	 * \param id the parameter id to configure
	 * \param flags additional flags
	 * \param param the parameter to configure
	 *
	 * \return 0 on success
	 *         -EINVAL when node is NULL
	 *         -ENOTSUP when there are no parameters implemented on \a node
	 *         -ENOENT the parameter is unknown
	 */
	int (*set_param) (void *object,
			  uint32_t id, uint32_t flags,
			  const struct spa_pod *param);

	/**
	 * Configure the given memory area with \a id on \a node. This
	 * structure is allocated by the host and is used to exchange
	 * data and parameters with the node.
	 *
	 * Setting an \a io of NULL will disable the node io.
	 *
	 * This function must be called from the main thread.
	 *
	 * \param id the id of the io area, the available ids can be
	 *        enumerated with the node parameters.
	 * \param data a io area memory
	 * \param size the size of \a data
	 * \return 0 on success
	 *         -EINVAL when invalid input is given
	 *         -ENOENT when \a id is unknown
	 *         -ENOSPC when \a size is too small
	 */
	int (*set_io) (void *object,
		       uint32_t id, void *data, size_t size);

	/**
	 * Send a command to a node.
	 *
	 * Upon completion, a command might change the state of a node.
	 *
	 * This function must be called from the main thread.
	 *
	 * \param node a  spa_node
	 * \param command a spa_command
	 * \return 0 on success
	 *         -EINVAL when node or command is NULL
	 *         -ENOTSUP when this node can't process commands
	 *         -EINVAL \a command is an invalid command
	 */
	int (*send_command) (void *object, const struct spa_command *command);

	/**
	 * Make a new port with \a port_id. The caller should use the lowest unused
	 * port id for the given \a direction.
	 *
	 * Port ids should be between 0 and max_ports as obtained from the info
	 * event.
	 *
	 * This function must be called from the main thread.
	 *
	 * \param node a  spa_node
	 * \param direction a enum \ref spa_direction
	 * \param port_id an unused port id
	 * \param props extra properties
	 * \return 0 on success
	 *         -EINVAL when node is NULL
	 */
	int (*add_port) (void *object,
			enum spa_direction direction, uint32_t port_id,
			const struct spa_dict *props);

	/**
	 * Remove a port with \a port_id.
	 *
	 * \param node a  spa_node
	 * \param direction a enum \ref spa_direction
	 * \param port_id a port id
	 * \return 0 on success
	 *         -EINVAL when node is NULL or when port_id is unknown or
	 *		when the port can't be removed.
	 */
	int (*remove_port) (void *object,
			enum spa_direction direction, uint32_t port_id);

	/**
	 * Enumerate all possible parameters of \a id on \a port_id of \a node
	 * that are compatible with \a filter.
	 *
	 * The result parameters can be queried and modified and ultimately be used
	 * to call port_set_param.
	 *
	 * The function will emit the result event up to \a max times with
	 * the result value. The seq in the result event will either be the
	 * \a seq number when executed synchronously or the async return
	 * value of this function when executed asynchronously.
	 *
	 * This function must be called from the main thread.
	 *
	 * \param node a spa_node
	 * \param seq a sequence number to pass to the result event when
	 *	this method is executed synchronously.
	 * \param direction an spa_direction
	 * \param port_id the port to query
	 * \param id the parameter id to query
	 * \param start the first index to query, 0 to get the first item
	 * \param max the maximum number of params to query
	 * \param filter a parameter filter or NULL for no filter
	 *
	 * \return 0 when no more items can be iterated.
	 *         -EINVAL when invalid parameters are given
	 *         -ENOENT when \a id is unknown
	 *         an async return value when the result event will be
	 *             emitted later.
	 */
	int (*port_enum_params) (void *object, int seq,
				 enum spa_direction direction, uint32_t port_id,
				 uint32_t id, uint32_t start, uint32_t max,
				 const struct spa_pod *filter);
	/**
	 * Set a parameter on \a port_id of \a node.
	 *
	 * When \a param is NULL, the parameter will be unset.
	 *
	 * This function must be called from the main thread. The node muse be paused
	 * or the port SPA_IO_Buffers area is NULL when this function is called with
	 * a param that changes the processing state (like a format change).
	 *
	 * \param node a struct \ref spa_node
	 * \param direction a enum \ref spa_direction
	 * \param port_id the port to configure
	 * \param id the parameter id to set
	 * \param flags optional flags
	 * \param param a struct \ref spa_pod with the parameter to set
	 * \return 0 on success
	 *         1 on success, the value of \a param might have been
	 *                changed depending on \a flags and the final value can be found by
	 *                doing port_enum_params.
	 *         -EINVAL when node is NULL or invalid arguments are given
	 *         -ESRCH when one of the mandatory param
	 *                 properties is not specified and SPA_NODE_PARAM_FLAG_FIXATE was
	 *                 not set in \a flags.
	 *         -ESRCH when the type or size of a property is not correct.
	 *         -ENOENT when the param id is not found
	 */
	int (*port_set_param) (void *object,
			       enum spa_direction direction,
			       uint32_t port_id,
			       uint32_t id, uint32_t flags,
			       const struct spa_pod *param);

	/**
	 * Tell the port to use the given buffers
	 *
	 * When \a flags contains SPA_NODE_BUFFERS_FLAG_ALLOC, the data
	 * in the buffers should point to an array of at least 1 data entry
	 * with the desired supported type that will be filled by this function.
	 *
	 * The port should also have a spa_io_buffers io area configured to exchange
	 * the buffers with the port.
	 *
	 * For an input port, all the buffers will remain dequeued.
	 * Once a buffer has been queued on a port in the spa_io_buffers,
	 * it should not be reused until the reuse_buffer callback is notified
	 * or when the buffer has been returned in the spa_io_buffers of
	 * the port.
	 *
	 * For output ports, all buffers will be queued in the port. When process
	 * returns SPA_STATUS_HAVE_DATA, buffers are available in one or more
	 * of the spa_io_buffers areas.
	 *
	 * When a buffer can be reused, port_reuse_buffer() should be called or the
	 * buffer_id should be placed in the spa_io_buffers area before calling
	 * process.
	 *
	 * Passing NULL as \a buffers will remove the reference that the port has
	 * on the buffers.
	 *
	 * When this function returns async, use the spa_node_sync operation to
	 * wait for completion.
	 *
	 * This function must be called from the main thread. The node muse be paused
	 * or the port SPA_IO_Buffers area is NULL when this function is called.
	 *
	 * \param object an object implementing the interface
	 * \param direction a port direction
	 * \param port_id a port id
	 * \param flags extra flags
	 * \param buffers an array of buffer pointers
	 * \param n_buffers number of elements in \a buffers
	 * \return 0 on success
	 */
	int (*port_use_buffers) (void *object,
				 enum spa_direction direction,
				 uint32_t port_id,
				 uint32_t flags,
				 struct spa_buffer **buffers,
				 uint32_t n_buffers);

	/**
	 * Configure the given memory area with \a id on \a port_id. This
	 * structure is allocated by the host and is used to exchange
	 * data and parameters with the port.
	 *
	 * Setting an \a io of NULL will disable the port io.
	 *
	 * This function must be called from the main thread.
	 *
	 * This function can be called when the node is running and the node
	 * must be prepared to handle changes in io areas while running. This
	 * is normally done by synchronizing the port io updates with the
	 * data processing loop.
	 *
	 * \param direction a spa_direction
	 * \param port_id a port id
	 * \param id the id of the io area, the available ids can be
	 *        enumerated with the port parameters.
	 * \param data a io area memory
	 * \param size the size of \a data
	 * \return 0 on success
	 *         -EINVAL when invalid input is given
	 *         -ENOENT when \a id is unknown
	 *         -ENOSPC when \a size is too small
	 */
	int (*port_set_io) (void *object,
			    enum spa_direction direction,
			    uint32_t port_id,
			    uint32_t id,
			    void *data, size_t size);

	/**
	 * Tell an output port to reuse a buffer.
	 *
	 * This function must be called from the data thread.
	 *
	 * \param node a spa_node
	 * \param port_id a port id
	 * \param buffer_id a buffer id to reuse
	 * \return 0 on success
	 *         -EINVAL when node is NULL
	 */
	int (*port_reuse_buffer) (void *object, uint32_t port_id, uint32_t buffer_id);

	/**
	 * Process the node
	 *
	 * This function must be called from the data thread.
	 *
	 * Output io areas with SPA_STATUS_NEED_DATA will recycle the
	 * buffers if any.
	 *
	 * Input areas with SPA_STATUS_HAVE_DATA are consumed if possible
	 * and the status is set to SPA_STATUS_NEED_DATA or SPA_STATUS_OK.
	 *
	 * When the node has new output buffers, the SPA_STATUS_HAVE_DATA
	 * bit will be set.
	 *
	 * When the node can accept new input in the next cycle, the
	 * SPA_STATUS_NEED_DATA bit will be set.
	 *
	 * Note that the node might return SPA_STATUS_NEED_DATA even when
	 * no input ports have this status. This means that the amount of
	 * data still available on the input ports is likely not going to
	 * be enough for the next cycle and the host might need to prefetch
	 * data for the next cycle.
	 */
	int (*process) (void *object);
};

#define spa_node_method(o,method,version,...)				\
({									\
	int _res = -ENOTSUP;						\
	struct spa_node *_n = o;					\
	spa_interface_call_res(&_n->iface,				\
			struct spa_node_methods, _res,			\
			method, version, ##__VA_ARGS__);		\
	_res;								\
})

#define spa_node_method_fast(o,method,version,...)			\
({									\
	int _res;							\
	struct spa_node *_n = o;					\
	spa_interface_call_fast_res(&_n->iface,				\
			struct spa_node_methods, _res,			\
			method, version, ##__VA_ARGS__);		\
	_res;								\
})

#define spa_node_add_listener(n,...)		spa_node_method(n, add_listener, 0, __VA_ARGS__)
#define spa_node_set_callbacks(n,...)		spa_node_method(n, set_callbacks, 0, __VA_ARGS__)
#define spa_node_sync(n,...)			spa_node_method(n, sync, 0, __VA_ARGS__)
#define spa_node_enum_params(n,...)		spa_node_method(n, enum_params, 0, __VA_ARGS__)
#define spa_node_set_param(n,...)		spa_node_method(n, set_param, 0, __VA_ARGS__)
#define spa_node_set_io(n,...)			spa_node_method(n, set_io, 0, __VA_ARGS__)
#define spa_node_send_command(n,...)		spa_node_method(n, send_command, 0, __VA_ARGS__)
#define spa_node_add_port(n,...)		spa_node_method(n, add_port, 0, __VA_ARGS__)
#define spa_node_remove_port(n,...)		spa_node_method(n, remove_port, 0, __VA_ARGS__)
#define spa_node_port_enum_params(n,...)	spa_node_method(n, port_enum_params, 0, __VA_ARGS__)
#define spa_node_port_set_param(n,...)		spa_node_method(n, port_set_param, 0, __VA_ARGS__)
#define spa_node_port_use_buffers(n,...)	spa_node_method(n, port_use_buffers, 0, __VA_ARGS__)
#define spa_node_port_set_io(n,...)		spa_node_method(n, port_set_io, 0, __VA_ARGS__)

#define spa_node_port_reuse_buffer(n,...)	spa_node_method(n, port_reuse_buffer, 0, __VA_ARGS__)
#define spa_node_port_reuse_buffer_fast(n,...)	spa_node_method_fast(n, port_reuse_buffer, 0, __VA_ARGS__)
#define spa_node_process(n)			spa_node_method(n, process, 0)
#define spa_node_process_fast(n)		spa_node_method_fast(n, process, 0)

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_NODE_H */
