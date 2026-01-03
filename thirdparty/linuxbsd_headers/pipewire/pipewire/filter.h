/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2019 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_FILTER_H
#define PIPEWIRE_FILTER_H

#ifdef __cplusplus
extern "C" {
#endif

/** \defgroup pw_filter Filter
 *
 * \brief PipeWire filter object class
 *
 * The filter object provides a convenient way to implement
 * processing filters.
 *
 * See also \ref api_pw_core
 */

/**
 * \addtogroup pw_filter
 * \{
 */
struct pw_filter;

#include <spa/buffer/buffer.h>
#include <spa/node/io.h>
#include <spa/param/param.h>
#include <spa/pod/command.h>

#include <pipewire/core.h>
#include <pipewire/stream.h>

/** \enum pw_filter_state The state of a filter  */
enum pw_filter_state {
	PW_FILTER_STATE_ERROR = -1,		/**< the stream is in error */
	PW_FILTER_STATE_UNCONNECTED = 0,	/**< unconnected */
	PW_FILTER_STATE_CONNECTING = 1,		/**< connection is in progress */
	PW_FILTER_STATE_PAUSED = 2,		/**< filter is connected and paused */
	PW_FILTER_STATE_STREAMING = 3		/**< filter is streaming */
};

#if 0
struct pw_buffer {
	struct spa_buffer *buffer;	/**< the spa buffer */
	void *user_data;		/**< user data attached to the buffer */
	uint64_t size;			/**< For input ports, this field is set by pw_filter
					  *  with the duration of the buffer in ticks.
					  *  For output ports, this field is set by the user.
					  *  This field is added for all queued buffers and
					  *  returned in the time info. */
};
#endif

/** Events for a filter. These events are always called from the mainloop
 * unless explicitly documented otherwise. */
struct pw_filter_events {
#define PW_VERSION_FILTER_EVENTS	1
	uint32_t version;

	void (*destroy) (void *data);
	/** when the filter state changes */
	void (*state_changed) (void *data, enum pw_filter_state old,
				enum pw_filter_state state, const char *error);

	/** when io changed on a port of the filter (when port_data is NULL). */
	void (*io_changed) (void *data, void *port_data,
			uint32_t id, void *area, uint32_t size);
	/** when a parameter changed on a port of the filter (when port_data is NULL). */
	void (*param_changed) (void *data, void *port_data,
			uint32_t id, const struct spa_pod *param);

        /** when a new buffer was created for a port */
        void (*add_buffer) (void *data, void *port_data, struct pw_buffer *buffer);
        /** when a buffer was destroyed for a port */
        void (*remove_buffer) (void *data, void *port_data, struct pw_buffer *buffer);

        /** do processing. This is normally called from the
	 *  mainloop but can also be called directly from the realtime data
	 *  thread if the user is prepared to deal with this. */
        void (*process) (void *data, struct spa_io_position *position);

	/** The filter is drained */
        void (*drained) (void *data);

	/** A command notify, Since 0.3.39:1 */
	void (*command) (void *data, const struct spa_command *command);
};

/** Convert a filter state to a readable string  */
const char * pw_filter_state_as_string(enum pw_filter_state state);

/** \enum pw_filter_flags Extra flags that can be used in \ref pw_filter_connect()  */
enum pw_filter_flags {
	PW_FILTER_FLAG_NONE = 0,			/**< no flags */
	PW_FILTER_FLAG_INACTIVE		= (1 << 0),	/**< start the filter inactive,
							  *  pw_filter_set_active() needs to be
							  *  called explicitly */
	PW_FILTER_FLAG_DRIVER		= (1 << 1),	/**< be a driver */
	PW_FILTER_FLAG_RT_PROCESS	= (1 << 2),	/**< call process from the realtime
							  *  thread */
	PW_FILTER_FLAG_CUSTOM_LATENCY	= (1 << 3),	/**< don't call the default latency algorithm
							  *  but emit the param_changed event for the
							  *  ports when Latency params are received. */
	PW_FILTER_FLAG_TRIGGER		= (1 << 4),	/**< the filter will not be scheduled
							  *  automatically but _trigger_process()
							  *  needs to be called. This can be used
							  *  when the filter depends on processing
							  *  of other filters. */
	PW_FILTER_FLAG_ASYNC		= (1 << 5),	/**< Buffers will not be dequeued/queued from
							  *  the realtime process() function. This is
							  *  assumed when RT_PROCESS is unset but can
							  *  also be the case when the process() function
							  *  does a trigger_process() that will then
							  *  dequeue/queue a buffer from another process()
							  *  function. since 0.3.73 */
};

enum pw_filter_port_flags {
	PW_FILTER_PORT_FLAG_NONE		= 0,		/**< no flags */
	PW_FILTER_PORT_FLAG_MAP_BUFFERS		= (1 << 0),	/**< mmap the buffers except DmaBuf */
	PW_FILTER_PORT_FLAG_ALLOC_BUFFERS	= (1 << 1),	/**< the application will allocate buffer
								  *  memory. In the add_buffer event, the
								  *  data of the buffer should be set */
};

/** Create a new unconneced \ref pw_filter
 * \return a newly allocated \ref pw_filter */
struct pw_filter *
pw_filter_new(struct pw_core *core,		/**< a \ref pw_core */
	      const char *name,			/**< a filter media name */
	      struct pw_properties *props	/**< filter properties, ownership is taken */);

struct pw_filter *
pw_filter_new_simple(struct pw_loop *loop,		/**< a \ref pw_loop to use */
		     const char *name,			/**< a filter media name */
		     struct pw_properties *props,	/**< filter properties, ownership is taken */
		     const struct pw_filter_events *events,	/**< filter events */
		     void *data					/**< data passed to events */);

/** Destroy a filter  */
void pw_filter_destroy(struct pw_filter *filter);

void pw_filter_add_listener(struct pw_filter *filter,
			    struct spa_hook *listener,
			    const struct pw_filter_events *events,
			    void *data);

enum pw_filter_state pw_filter_get_state(struct pw_filter *filter, const char **error);

const char *pw_filter_get_name(struct pw_filter *filter);

struct pw_core *pw_filter_get_core(struct pw_filter *filter);

/** Connect a filter for processing.
 * \return 0 on success < 0 on error.
 *
 * You should connect to the process event and use pw_filter_dequeue_buffer()
 * to get the latest metadata and data. */
int
pw_filter_connect(struct pw_filter *filter,		/**< a \ref pw_filter */
		  enum pw_filter_flags flags,		/**< filter flags */
		  const struct spa_pod **params,	/**< an array with params. */
		  uint32_t n_params			/**< number of items in \a params */);

/** Get the node ID of the filter.
 * \return node ID. */
uint32_t
pw_filter_get_node_id(struct pw_filter *filter);

/** Disconnect \a filter  */
int pw_filter_disconnect(struct pw_filter *filter);

/** add a port to the filter, returns user data of port_data_size. */
void *pw_filter_add_port(struct pw_filter *filter,	/**< a \ref pw_filter */
		enum pw_direction direction,		/**< port direction */
		enum pw_filter_port_flags flags,	/**< port flags */
		size_t port_data_size,			/**< allocated and given to the user as port_data */
		struct pw_properties *props,		/**< port properties, ownership is taken */
		const struct spa_pod **params,		/**< an array of params. The params should
							  *  ideally contain the supported formats */
		uint32_t n_params			/**< number of elements in \a params */);

/** remove a port from the filter */
int pw_filter_remove_port(void *port_data		/**< data associated with port */);

/** get properties, port_data of NULL will give global properties */
const struct pw_properties *pw_filter_get_properties(struct pw_filter *filter,
		void *port_data);

/** Update properties, use NULL port_data for global filter properties */
int pw_filter_update_properties(struct pw_filter *filter,
		void *port_data, const struct spa_dict *dict);

/** Set the filter in error state */
int pw_filter_set_error(struct pw_filter *filter,	/**< a \ref pw_filter */
			int res,			/**< a result code */
			const char *error,		/**< an error message */
			...
			) SPA_PRINTF_FUNC(3, 4);

/** Update params, use NULL port_data for global filter params */
int
pw_filter_update_params(struct pw_filter *filter,	/**< a \ref pw_filter */
			void *port_data,		/**< data associated with port */
			const struct spa_pod **params,	/**< an array of params. */
			uint32_t n_params		/**< number of elements in \a params */);


/** Query the time on the filter, deprecated, use the spa_io_position in the
 * process() method for timing information. */
SPA_DEPRECATED
int pw_filter_get_time(struct pw_filter *filter, struct pw_time *time);

/** Get a buffer that can be filled for output ports or consumed
 * for input ports.  */
struct pw_buffer *pw_filter_dequeue_buffer(void *port_data);

/** Submit a buffer for playback or recycle a buffer for capture. */
int pw_filter_queue_buffer(void *port_data, struct pw_buffer *buffer);

/** Get a data pointer to the buffer data */
void *pw_filter_get_dsp_buffer(void *port_data, uint32_t n_samples);

/** Activate or deactivate the filter  */
int pw_filter_set_active(struct pw_filter *filter, bool active);

/** Flush a filter. When \a drain is true, the drained callback will
 * be called when all data is played or recorded */
int pw_filter_flush(struct pw_filter *filter, bool drain);

/** Check if the filter is driving. The filter needs to have the
 * PW_FILTER_FLAG_DRIVER set. When the filter is driving,
 * pw_filter_trigger_process() needs to be called when data is
 * available (output) or needed (input). Since 0.3.66 */
bool pw_filter_is_driving(struct pw_filter *filter);

/** Trigger a push/pull on the filter. One iteration of the graph will
 * be scheduled and process() will be called. Since 0.3.66 */
int pw_filter_trigger_process(struct pw_filter *filter);

/**
 * \}
 */

#ifdef __cplusplus
}
#endif

#endif /* PIPEWIRE_FILTER_H */
