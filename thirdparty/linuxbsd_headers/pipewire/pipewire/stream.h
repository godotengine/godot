/* PipeWire */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef PIPEWIRE_STREAM_H
#define PIPEWIRE_STREAM_H

#ifdef __cplusplus
extern "C" {
#endif

/** \page page_streams Streams
 *
 * \see \ref pw_stream
 *
 * \section sec_overview Overview
 *
 * \ref pw_stream "Streams" are used to exchange data with the
 * PipeWire server. A stream is a wrapper around a proxy for a pw_client_node
 * with an adapter. This means the stream will automatically do conversion
 * to the type required by the server.
 *
 * Streams can be used to:
 *
 * \li Consume a stream from PipeWire. This is a PW_DIRECTION_INPUT stream.
 * \li Produce a stream to PipeWire. This is a PW_DIRECTION_OUTPUT stream
 *
 * You can connect the stream port to a specific server port or let PipeWire
 * choose a port for you.
 *
 * For more complicated nodes such as filters or ports with multiple
 * inputs and/or outputs you will need to use the pw_filter or make
 * a pw_node yourself and export it with \ref pw_core_export.
 *
 * Streams can also be used to:
 *
 * \li Implement a Sink in PipeWire. This is a PW_DIRECTION_INPUT stream.
 * \li Implement a Source in PipeWire. This is a PW_DIRECTION_OUTPUT stream
 *
 * In this case, the PW_KEY_MEDIA_CLASS property needs to be set to
 * "Audio/Sink" or "Audio/Source" respectively.
 *
 * \section sec_create Create
 *
 * Make a new stream with \ref pw_stream_new(). You will need to specify
 * a name for the stream and extra properties. The basic set of properties
 * each stream must provide is filled in automatically.
 *
 * Once the stream is created, the state_changed event should be used to
 * track the state of the stream.
 *
 * \section sec_connect Connect
 *
 * The stream is initially unconnected. To connect the stream, use
 * \ref pw_stream_connect(). Pass the desired direction as an argument.
 *
 * The direction is:

 * \li PW_DIRECTION_INPUT for a stream that *consumes* data. This can be a
 * stream that captures from a Source or a when the stream is used to
 * implement a Sink.
 *
 * \li PW_DIRECTION_OUTPUT for a stream that *produces* data. This can be a
 * stream that plays to a Sink or when the stream is used to implement
 * a Source.
 *
 * \subsection ssec_stream_target Stream target
 *
 * To make the newly connected stream automatically connect to an existing
 * PipeWire node, use the \ref PW_STREAM_FLAG_AUTOCONNECT and set the
 * PW_KEY_OBJECT_SERIAL or the PW_KEY_NODE_NAME value of the target node
 * in the PW_KEY_TARGET_OBJECT property before connecting.
 *
 * \subsection ssec_stream_formats Stream formats
 *
 * An array of possible formats that this stream can consume or provide
 * must be specified.
 *
 * \section sec_format Format negotiation
 *
 * After connecting the stream, the server will want to configure some
 * parameters on the stream. You will be notified of these changes
 * with the param_changed event.
 *
 * When a format param change is emitted, the client should now prepare
 * itself to deal with the format and complete the negotiation procedure
 * with a call to \ref pw_stream_update_params().
 *
 * As arguments to \ref pw_stream_update_params() an array of spa_param
 * structures must be given. They contain parameters such as buffer size,
 * number of buffers, required metadata and other parameters for the
 * media buffers.
 *
 * \section sec_buffers Buffer negotiation
 *
 * After completing the format negotiation, PipeWire will allocate and
 * notify the stream of the buffers that will be used to exchange data
 * between client and server.
 *
 * With the add_buffer event, a stream will be notified of a new buffer
 * that can be used for data transport. You can attach user_data to these
 * buffers. The buffers can only be used with the stream that emitted
 * the add_buffer event.
 *
 * After the buffers are negotiated, the stream will transition to the
 * \ref PW_STREAM_STATE_PAUSED state.
 *
 * \section sec_streaming Streaming
 *
 * From the \ref PW_STREAM_STATE_PAUSED state, the stream can be set to
 * the \ref PW_STREAM_STATE_STREAMING state by the PipeWire server when
 * data transport is started.
 *
 * Depending on how the stream was connected it will need to Produce or
 * Consume data for/from PipeWire as explained in the following
 * subsections.
 *
 * \subsection ssec_consume Consume data
 *
 * The process event is emitted for each new buffer that can be
 * consumed.
 *
 * \ref pw_stream_dequeue_buffer() should be used to get the data and
 * metadata of the buffer.
 *
 * The buffer is owned by the stream and stays alive until the
 * remove_buffer callback has returned or the stream is destroyed.
 *
 * When the buffer has been processed, call \ref pw_stream_queue_buffer()
 * to let PipeWire reuse the buffer.
 *
 * \subsection ssec_produce Produce data
 *
 * \ref pw_stream_dequeue_buffer() gives an empty buffer that can be filled.
 *
 * The buffer is owned by the stream and stays alive until the
 * remove_buffer event is emitted or the stream is destroyed.
 *
 * Filled buffers should be queued with \ref pw_stream_queue_buffer().
 *
 * The process event is emitted when PipeWire has emptied a buffer that
 * can now be refilled.
 *
 * \section sec_stream_disconnect Disconnect
 *
 * Use \ref pw_stream_disconnect() to disconnect a stream after use.
 *
 * \section sec_stream_configuration Configuration
 *
 * \subsection ssec_config_properties Stream Properties
 *
 * \subsection ssec_config_rules Stream Rules
 *
 * \section sec_stream_environment Environment Variables
 *
 * The environment variable PIPEWIRE_AUTOCONNECT can be used to override the
 * flag and force apps to autoconnect or not.
 *
 */
/** \defgroup pw_stream Stream
 *
 * \brief PipeWire stream objects
 *
 * The stream object provides a convenient way to send and
 * receive data streams from/to PipeWire.
 *
 * \see \ref page_streams, \ref api_pw_core
 */

/**
 * \addtogroup pw_stream
 * \{
 */
struct pw_stream;

#include <spa/buffer/buffer.h>
#include <spa/param/param.h>
#include <spa/pod/command.h>

/** \enum pw_stream_state The state of a stream */
enum pw_stream_state {
	PW_STREAM_STATE_ERROR = -1,		/**< the stream is in error */
	PW_STREAM_STATE_UNCONNECTED = 0,	/**< unconnected */
	PW_STREAM_STATE_CONNECTING = 1,		/**< connection is in progress */
	PW_STREAM_STATE_PAUSED = 2,		/**< paused */
	PW_STREAM_STATE_STREAMING = 3		/**< streaming */
};

/** a buffer structure obtained from pw_stream_dequeue_buffer(). The size of this
  * structure can grow as more field are added in the future */
struct pw_buffer {
	struct spa_buffer *buffer;	/**< the spa buffer */
	void *user_data;		/**< user data attached to the buffer */
	uint64_t size;			/**< This field is set by the user and the sum of
					  *  all queued buffer is returned in the time info.
					  *  For audio, it is advised to use the number of
					  *  samples in the buffer for this field. */
	uint64_t requested;		/**< For playback streams, this field contains the
					  *  suggested amount of data to provide. For audio
					  *  streams this will be the amount of samples
					  *  required by the resampler. This field is 0
					  *  when no suggestion is provided. Since 0.3.49 */
};

struct pw_stream_control {
	const char *name;		/**< name of the control */
	uint32_t flags;			/**< extra flags (unused) */
	float def;			/**< default value */
	float min;			/**< min value */
	float max;			/**< max value */
	float *values;			/**< array of values */
	uint32_t n_values;		/**< number of values in array */
	uint32_t max_values;		/**< max values that can be set on this control */
};

/** A time structure.
 *
 * Use pw_stream_get_time_n() to get an updated time snapshot of the stream.
 * The time snapshot can give information about the time in the driver of the
 * graph, the delay to the edge of the graph and the internal queuing in the
 * stream.
 *
 * pw_time.ticks gives a monotonic increasing counter of the time in the graph
 * driver. I can be used to generate a timetime to schedule samples as well
 * as detect discontinuities in the timeline caused by xruns.
 *
 * pw_time.delay is expressed as pw_time.rate, the time domain of the graph. This
 * value, and pw_time.ticks, were captured at pw_time.now and can be extrapolated
 * to the current time like this:
 *
 *\code{.c}
 *    struct timespec ts;
 *    clock_gettime(CLOCK_MONOTONIC, &ts);
 *    int64_t diff = SPA_TIMESPEC_TO_NSEC(&ts) - pw_time.now;
 *    int64_t elapsed = (pw_time.rate.denom * diff) / (pw_time.rate.num * SPA_NSEC_PER_SEC);
 *\endcode
 *
 * pw_time.delay contains the total delay that a signal will travel through the
 * graph. This includes the delay caused by filters in the graph as well as delays
 * caused by the hardware. The delay is usually quite stable and should only change when
 * the topology, quantum or samplerate of the graph changes.
 *
 * pw_time.queued and pw_time.buffered is expressed in the time domain of the stream,
 * or the format that is used for the buffers of this stream.
 *
 * pw_time.queued is the sum of all the pw_buffer.size fields of the buffers that are
 * currently queued in the stream but not yet processed. The application can choose
 * the units of this value, for example, time, samples or bytes (below expressed
 * as app.rate).
 *
 * pw_time.buffered is format dependent, for audio/raw it contains the number of samples
 * that are buffered inside the resampler/converter.
 *
 * The total delay of data in a stream is the sum of the queued and buffered data
 * (not yet processed data) and the delay to the edge of the graph, usually a
 * playback or capture device.
 *
 * For an audio playback stream, if you were to queue a buffer, the total delay
 * in milliseconds for the first sample in the newly queued buffer to be played
 * by the hardware can be calculated as:
 *
 *\code{.unparsed}
 *  (pw_time.buffered * 1000 / stream.samplerate) +
 *    (pw_time.queued * 1000 / app.rate) +
 *     ((pw_time.delay - elapsed) * 1000 * pw_time.rate.num / pw_time.rate.denom)
 *\endcode
 *
 * The current extrapolated time (in ms) in the source or sink can be calculated as:
 *
 *\code{.unparsed}
 *  (pw_time.ticks + elapsed) * 1000 * pw_time.rate.num / pw_time.rate.denom
 *\endcode
 *
 * Below is an overview of the different timing values:
 *
 *\code{.unparsed}
 *           stream time domain           graph time domain
 *         /-----------------------\/-----------------------------\
 *
 * queue     +-+ +-+  +-----------+                 +--------+
 * ---->     | | | |->| converter | ->   graph  ->  | kernel | -> speaker
 * <----     +-+ +-+  +-----------+                 +--------+
 * dequeue   buffers                \-------------------/\--------/
 *                                     graph              internal
 *                                    latency             latency
 *         \--------/\-------------/\-----------------------------/
 *           queued      buffered            delay
 *\endcode
 */
struct pw_time {
	int64_t now;			/**< the monotonic time in nanoseconds. This is the time
					  *  when this time report was updated. It is usually
					  *  updated every graph cycle. You can use the current
					  *  monotonic time to calculate the elapsed time between
					  *  this report and the current state and calculate
					  *  updated ticks and delay values. */
	struct spa_fraction rate;	/**< the rate of \a ticks and delay. This is usually
					  *  expressed in 1/<samplerate>. */
	uint64_t ticks;			/**< the ticks at \a now. This is the current time that
					  *  the remote end is reading/writing. This is monotonicaly
					  *  increasing. */
	int64_t delay;			/**< delay to device. This is the time it will take for
					  *  the next output sample of the stream to be presented by
					  *  the playback device or the time a sample traveled
					  *  from the capture device. This delay includes the
					  *  delay introduced by all filters on the path between
					  *  the stream and the device. The delay is normally
					  *  constant in a graph and can change when the topology
					  *  of the graph or the quantum changes. This delay does
					  *  not include the delay caused by queued buffers. */
	uint64_t queued;		/**< data queued in the stream, this is the sum
					  *  of the size fields in the pw_buffer that are
					  *  currently queued */
	uint64_t buffered;		/**< for audio/raw streams, this contains the extra
					  *  number of samples buffered in the resampler.
					  *  Since 0.3.50. */
	uint32_t queued_buffers;	/**< The number of buffers that are queued. Since 0.3.50 */
	uint32_t avail_buffers;		/**< The number of buffers that can be dequeued. Since 0.3.50 */
};

#include <pipewire/port.h>

/** Events for a stream. These events are always called from the mainloop
 * unless explicitly documented otherwise. */
struct pw_stream_events {
#define PW_VERSION_STREAM_EVENTS	2
	uint32_t version;

	void (*destroy) (void *data);
	/** when the stream state changes */
	void (*state_changed) (void *data, enum pw_stream_state old,
				enum pw_stream_state state, const char *error);

	/** Notify information about a control.  */
	void (*control_info) (void *data, uint32_t id, const struct pw_stream_control *control);

	/** when io changed on the stream. */
	void (*io_changed) (void *data, uint32_t id, void *area, uint32_t size);
	/** when a parameter changed */
	void (*param_changed) (void *data, uint32_t id, const struct spa_pod *param);

        /** when a new buffer was created for this stream */
        void (*add_buffer) (void *data, struct pw_buffer *buffer);
        /** when a buffer was destroyed for this stream */
        void (*remove_buffer) (void *data, struct pw_buffer *buffer);

        /** when a buffer can be queued (for playback streams) or
         *  dequeued (for capture streams). This is normally called from the
	 *  mainloop but can also be called directly from the realtime data
	 *  thread if the user is prepared to deal with this. */
        void (*process) (void *data);

	/** The stream is drained */
        void (*drained) (void *data);

	/** A command notify, Since 0.3.39:1 */
	void (*command) (void *data, const struct spa_command *command);

	/** a trigger_process completed. Since version 0.3.40:2 */
	void (*trigger_done) (void *data);
};

/** Convert a stream state to a readable string */
const char * pw_stream_state_as_string(enum pw_stream_state state);

/** \enum pw_stream_flags Extra flags that can be used in \ref pw_stream_connect() */
enum pw_stream_flags {
	PW_STREAM_FLAG_NONE = 0,			/**< no flags */
	PW_STREAM_FLAG_AUTOCONNECT	= (1 << 0),	/**< try to automatically connect
							  *  this stream */
	PW_STREAM_FLAG_INACTIVE		= (1 << 1),	/**< start the stream inactive,
							  *  pw_stream_set_active() needs to be
							  *  called explicitly */
	PW_STREAM_FLAG_MAP_BUFFERS	= (1 << 2),	/**< mmap the buffers except DmaBuf */
	PW_STREAM_FLAG_DRIVER		= (1 << 3),	/**< be a driver */
	PW_STREAM_FLAG_RT_PROCESS	= (1 << 4),	/**< call process from the realtime
							  *  thread. You MUST use RT safe functions
							  *  in the process callback. */
	PW_STREAM_FLAG_NO_CONVERT	= (1 << 5),	/**< don't convert format */
	PW_STREAM_FLAG_EXCLUSIVE	= (1 << 6),	/**< require exclusive access to the
							  *  device */
	PW_STREAM_FLAG_DONT_RECONNECT	= (1 << 7),	/**< don't try to reconnect this stream
							  *  when the sink/source is removed */
	PW_STREAM_FLAG_ALLOC_BUFFERS	= (1 << 8),	/**< the application will allocate buffer
							  *  memory. In the add_buffer event, the
							  *  data of the buffer should be set */
	PW_STREAM_FLAG_TRIGGER		= (1 << 9),	/**< the output stream will not be scheduled
							  *  automatically but _trigger_process()
							  *  needs to be called. This can be used
							  *  when the output of the stream depends
							  *  on input from other streams. */
	PW_STREAM_FLAG_ASYNC		= (1 << 10),	/**< Buffers will not be dequeued/queued from
							  *  the realtime process() function. This is
							  *  assumed when RT_PROCESS is unset but can
							  *  also be the case when the process() function
							  *  does a trigger_process() that will then
							  *  dequeue/queue a buffer from another process()
							  *  function. since 0.3.73 */
	PW_STREAM_FLAG_EARLY_PROCESS	= (1 << 11),	/**< Call process as soon as there is a buffer
							  *  to dequeue. This is only relevant for
							  *  playback and when not using RT_PROCESS. It
							  *  can be used to keep the maximum number of
							  *  buffers queued. Since 0.3.81 */
};

/** Create a new unconneced \ref pw_stream
 * \return a newly allocated \ref pw_stream */
struct pw_stream *
pw_stream_new(struct pw_core *core,		/**< a \ref pw_core */
	      const char *name,			/**< a stream media name */
	      struct pw_properties *props	/**< stream properties, ownership is taken */);

struct pw_stream *
pw_stream_new_simple(struct pw_loop *loop,	/**< a \ref pw_loop to use */
		     const char *name,		/**< a stream media name */
		     struct pw_properties *props,/**< stream properties, ownership is taken */
		     const struct pw_stream_events *events,	/**< stream events */
		     void *data					/**< data passed to events */);

/** Destroy a stream */
void pw_stream_destroy(struct pw_stream *stream);

void pw_stream_add_listener(struct pw_stream *stream,
			    struct spa_hook *listener,
			    const struct pw_stream_events *events,
			    void *data);

enum pw_stream_state pw_stream_get_state(struct pw_stream *stream, const char **error);

const char *pw_stream_get_name(struct pw_stream *stream);

struct pw_core *pw_stream_get_core(struct pw_stream *stream);

const struct pw_properties *pw_stream_get_properties(struct pw_stream *stream);

int pw_stream_update_properties(struct pw_stream *stream, const struct spa_dict *dict);

/** Connect a stream for input or output on \a port_path.
 * \return 0 on success < 0 on error.
 *
 * You should connect to the process event and use pw_stream_dequeue_buffer()
 * to get the latest metadata and data. */
int
pw_stream_connect(struct pw_stream *stream,		/**< a \ref pw_stream */
		  enum pw_direction direction,		/**< the stream direction */
		  uint32_t target_id,			/**< should have the value PW_ID_ANY.
							  * To select a specific target
							  * node, specify the
							  * PW_KEY_OBJECT_SERIAL or the
							  * PW_KEY_NODE_NAME value of the target
							  * node in the PW_KEY_TARGET_OBJECT
							  * property of the stream.
							  * Specifying target nodes by
							  * their id is deprecated.
							  */
		  enum pw_stream_flags flags,		/**< stream flags */
		  const struct spa_pod **params,	/**< an array with params. The params
							  *  should ideally contain supported
							  *  formats. */
		  uint32_t n_params			/**< number of items in \a params */);

/** Get the node ID of the stream.
 * \return node ID. */
uint32_t
pw_stream_get_node_id(struct pw_stream *stream);

/** Disconnect \a stream  */
int pw_stream_disconnect(struct pw_stream *stream);

/** Set the stream in error state */
int pw_stream_set_error(struct pw_stream *stream,	/**< a \ref pw_stream */
			int res,			/**< a result code */
			const char *error,		/**< an error message */
			...) SPA_PRINTF_FUNC(3, 4);

/** Update the param exposed on the stream. */
int
pw_stream_update_params(struct pw_stream *stream,	/**< a \ref pw_stream */
			const struct spa_pod **params,	/**< an array of params. */
			uint32_t n_params		/**< number of elements in \a params */);

/**
 * Set a parameter on the stream. This is like pw_stream_set_control() but with
 * a complete spa_pod param. It can also be called from the param_changed event handler
 * to intercept and modify the param for the adapter. Since 0.3.70 */
int pw_stream_set_param(struct pw_stream *stream,	/**< a \ref pw_stream */
			uint32_t id,			/**< the id of the param */
			const struct spa_pod *param	/**< the params to set */);

/** Get control values */
const struct pw_stream_control *pw_stream_get_control(struct pw_stream *stream, uint32_t id);

/** Set control values */
int pw_stream_set_control(struct pw_stream *stream, uint32_t id, uint32_t n_values, float *values, ...);

/** Query the time on the stream */
int pw_stream_get_time_n(struct pw_stream *stream, struct pw_time *time, size_t size);

/** Query the time on the stream, deprecated since 0.3.50,
 * use pw_stream_get_time_n() to get the fields added since 0.3.50. */
SPA_DEPRECATED
int pw_stream_get_time(struct pw_stream *stream, struct pw_time *time);

/** Get a buffer that can be filled for playback streams or consumed
 * for capture streams. */
struct pw_buffer *pw_stream_dequeue_buffer(struct pw_stream *stream);

/** Submit a buffer for playback or recycle a buffer for capture. */
int pw_stream_queue_buffer(struct pw_stream *stream, struct pw_buffer *buffer);

/** Activate or deactivate the stream */
int pw_stream_set_active(struct pw_stream *stream, bool active);

/** Flush a stream. When \a drain is true, the drained callback will
 * be called when all data is played or recorded */
int pw_stream_flush(struct pw_stream *stream, bool drain);

/** Check if the stream is driving. The stream needs to have the
 * PW_STREAM_FLAG_DRIVER set. When the stream is driving,
 * pw_stream_trigger_process() needs to be called when data is
 * available (output) or needed (input). Since 0.3.34 */
bool pw_stream_is_driving(struct pw_stream *stream);

/** Trigger a push/pull on the stream. One iteration of the graph will
 * scheduled and process() will be called. Since 0.3.34 */
int pw_stream_trigger_process(struct pw_stream *stream);

/**
 * \}
 */

#ifdef __cplusplus
}
#endif

#endif /* PIPEWIRE_STREAM_H */
