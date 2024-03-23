#ifndef foodefhfoo
#define foodefhfoo

/***
  This file is part of PulseAudio.

  Copyright 2004-2006 Lennart Poettering
  Copyright 2006 Pierre Ossman <ossman@cendio.se> for Cendio AB

  PulseAudio is free software; you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as
  published by the Free Software Foundation; either version 2.1 of the
  License, or (at your option) any later version.

  PulseAudio is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with PulseAudio; if not, see <http://www.gnu.org/licenses/>.
***/

#include <inttypes.h>
#include <sys/time.h>

#include <pulse/cdecl.h>
#include <pulse/sample.h>
#include <pulse/version.h>

/** \file
 * Global definitions */

PA_C_DECL_BEGIN

/** The state of a connection context */
typedef enum pa_context_state {
    PA_CONTEXT_UNCONNECTED,    /**< The context hasn't been connected yet */
    PA_CONTEXT_CONNECTING,     /**< A connection is being established */
    PA_CONTEXT_AUTHORIZING,    /**< The client is authorizing itself to the daemon */
    PA_CONTEXT_SETTING_NAME,   /**< The client is passing its application name to the daemon */
    PA_CONTEXT_READY,          /**< The connection is established, the context is ready to execute operations */
    PA_CONTEXT_FAILED,         /**< The connection failed or was disconnected */
    PA_CONTEXT_TERMINATED      /**< The connection was terminated cleanly */
} pa_context_state_t;

/** Return non-zero if the passed state is one of the connected states. \since 0.9.11 */
static inline int PA_CONTEXT_IS_GOOD(pa_context_state_t x) {
    return
        x == PA_CONTEXT_CONNECTING ||
        x == PA_CONTEXT_AUTHORIZING ||
        x == PA_CONTEXT_SETTING_NAME ||
        x == PA_CONTEXT_READY;
}

/** \cond fulldocs */
#define PA_CONTEXT_UNCONNECTED PA_CONTEXT_UNCONNECTED
#define PA_CONTEXT_CONNECTING PA_CONTEXT_CONNECTING
#define PA_CONTEXT_AUTHORIZING PA_CONTEXT_AUTHORIZING
#define PA_CONTEXT_SETTING_NAME PA_CONTEXT_SETTING_NAME
#define PA_CONTEXT_READY PA_CONTEXT_READY
#define PA_CONTEXT_FAILED PA_CONTEXT_FAILED
#define PA_CONTEXT_TERMINATED PA_CONTEXT_TERMINATED
#define PA_CONTEXT_IS_GOOD PA_CONTEXT_IS_GOOD
/** \endcond */

/** The state of a stream */
typedef enum pa_stream_state {
    PA_STREAM_UNCONNECTED,  /**< The stream is not yet connected to any sink or source */
    PA_STREAM_CREATING,     /**< The stream is being created */
    PA_STREAM_READY,        /**< The stream is established, you may pass audio data to it now */
    PA_STREAM_FAILED,       /**< An error occurred that made the stream invalid */
    PA_STREAM_TERMINATED    /**< The stream has been terminated cleanly */
} pa_stream_state_t;

/** Return non-zero if the passed state is one of the connected states. \since 0.9.11 */
static inline int PA_STREAM_IS_GOOD(pa_stream_state_t x) {
    return
        x == PA_STREAM_CREATING ||
        x == PA_STREAM_READY;
}

/** \cond fulldocs */
#define PA_STREAM_UNCONNECTED PA_STREAM_UNCONNECTED
#define PA_STREAM_CREATING PA_STREAM_CREATING
#define PA_STREAM_READY PA_STREAM_READY
#define PA_STREAM_FAILED PA_STREAM_FAILED
#define PA_STREAM_TERMINATED PA_STREAM_TERMINATED
#define PA_STREAM_IS_GOOD PA_STREAM_IS_GOOD
/** \endcond */

/** The state of an operation */
typedef enum pa_operation_state {
    PA_OPERATION_RUNNING,
    /**< The operation is still running */
    PA_OPERATION_DONE,
    /**< The operation has completed */
    PA_OPERATION_CANCELLED
    /**< The operation has been cancelled. Operations may get cancelled by the
     * application, or as a result of the context getting disconneted while the
     * operation is pending. */
} pa_operation_state_t;

/** \cond fulldocs */
#define PA_OPERATION_RUNNING PA_OPERATION_RUNNING
#define PA_OPERATION_DONE PA_OPERATION_DONE
#define PA_OPERATION_CANCELED PA_OPERATION_CANCELLED
#define PA_OPERATION_CANCELLED PA_OPERATION_CANCELLED
/** \endcond */

/** An invalid index */
#define PA_INVALID_INDEX ((uint32_t) -1)

/** Some special flags for contexts. */
typedef enum pa_context_flags {
    PA_CONTEXT_NOFLAGS = 0x0000U,
    /**< Flag to pass when no specific options are needed (used to avoid casting)  \since 0.9.19 */
    PA_CONTEXT_NOAUTOSPAWN = 0x0001U,
    /**< Disabled autospawning of the PulseAudio daemon if required */
    PA_CONTEXT_NOFAIL = 0x0002U
    /**< Don't fail if the daemon is not available when pa_context_connect() is called, instead enter PA_CONTEXT_CONNECTING state and wait for the daemon to appear.  \since 0.9.15 */
} pa_context_flags_t;

/** \cond fulldocs */
/* Allow clients to check with #ifdef for those flags */
#define PA_CONTEXT_NOAUTOSPAWN PA_CONTEXT_NOAUTOSPAWN
#define PA_CONTEXT_NOFAIL PA_CONTEXT_NOFAIL
/** \endcond */

/** Direction bitfield - while we currently do not expose anything bidirectional,
  one should test against the bit instead of the value (e.g.\ if (d & PA_DIRECTION_OUTPUT)),
  because we might add bidirectional stuff in the future. \since 2.0
*/
typedef enum pa_direction {
    PA_DIRECTION_OUTPUT = 0x0001U,  /**< Output direction */
    PA_DIRECTION_INPUT = 0x0002U    /**< Input direction */
} pa_direction_t;

/** \cond fulldocs */
#define PA_DIRECTION_OUTPUT PA_DIRECTION_OUTPUT
#define PA_DIRECTION_INPUT PA_DIRECTION_INPUT
/** \endcond */

/** The type of device we are dealing with */
typedef enum pa_device_type {
    PA_DEVICE_TYPE_SINK,     /**< Playback device */
    PA_DEVICE_TYPE_SOURCE    /**< Recording device */
} pa_device_type_t;

/** \cond fulldocs */
#define PA_DEVICE_TYPE_SINK PA_DEVICE_TYPE_SINK
#define PA_DEVICE_TYPE_SOURCE PA_DEVICE_TYPE_SOURCE
/** \endcond */

/** The direction of a pa_stream object */
typedef enum pa_stream_direction {
    PA_STREAM_NODIRECTION,   /**< Invalid direction */
    PA_STREAM_PLAYBACK,      /**< Playback stream */
    PA_STREAM_RECORD,        /**< Record stream */
    PA_STREAM_UPLOAD         /**< Sample upload stream */
} pa_stream_direction_t;

/** \cond fulldocs */
#define PA_STREAM_NODIRECTION PA_STREAM_NODIRECTION
#define PA_STREAM_PLAYBACK PA_STREAM_PLAYBACK
#define PA_STREAM_RECORD PA_STREAM_RECORD
#define PA_STREAM_UPLOAD PA_STREAM_UPLOAD
/** \endcond */

/** Some special flags for stream connections. */
typedef enum pa_stream_flags {

    PA_STREAM_NOFLAGS = 0x0000U,
    /**< Flag to pass when no specific options are needed (used to avoid casting)  \since 0.9.19 */

    PA_STREAM_START_CORKED = 0x0001U,
    /**< Create the stream corked, requiring an explicit
     * pa_stream_cork() call to uncork it. */

    PA_STREAM_INTERPOLATE_TIMING = 0x0002U,
    /**< Interpolate the latency for this stream. When enabled,
     * pa_stream_get_latency() and pa_stream_get_time() will try to
     * estimate the current record/playback time based on the local
     * time that passed since the last timing info update.  Using this
     * option has the advantage of not requiring a whole roundtrip
     * when the current playback/recording time is needed. Consider
     * using this option when requesting latency information
     * frequently. This is especially useful on long latency network
     * connections. It makes a lot of sense to combine this option
     * with PA_STREAM_AUTO_TIMING_UPDATE. */

    PA_STREAM_NOT_MONOTONIC = 0x0004U,
    /**< Don't force the time to increase monotonically. If this
     * option is enabled, pa_stream_get_time() will not necessarily
     * return always monotonically increasing time values on each
     * call. This may confuse applications which cannot deal with time
     * going 'backwards', but has the advantage that bad transport
     * latency estimations that caused the time to jump ahead can
     * be corrected quickly, without the need to wait. (Please note
     * that this flag was named PA_STREAM_NOT_MONOTONOUS in releases
     * prior to 0.9.11. The old name is still defined too, for
     * compatibility reasons. */

    PA_STREAM_AUTO_TIMING_UPDATE = 0x0008U,
    /**< If set timing update requests are issued periodically
     * automatically. Combined with PA_STREAM_INTERPOLATE_TIMING you
     * will be able to query the current time and latency with
     * pa_stream_get_time() and pa_stream_get_latency() at all times
     * without a packet round trip.*/

    PA_STREAM_NO_REMAP_CHANNELS = 0x0010U,
    /**< Don't remap channels by their name, instead map them simply
     * by their index. Implies PA_STREAM_NO_REMIX_CHANNELS. Only
     * supported when the server is at least PA 0.9.8. It is ignored
     * on older servers.\since 0.9.8 */

    PA_STREAM_NO_REMIX_CHANNELS = 0x0020U,
    /**< When remapping channels by name, don't upmix or downmix them
     * to related channels. Copy them into matching channels of the
     * device 1:1. Only supported when the server is at least PA
     * 0.9.8. It is ignored on older servers. \since 0.9.8 */

    PA_STREAM_FIX_FORMAT = 0x0040U,
    /**< Use the sample format of the sink/device this stream is being
     * connected to, and possibly ignore the format the sample spec
     * contains -- but you still have to pass a valid value in it as a
     * hint to PulseAudio what would suit your stream best. If this is
     * used you should query the used sample format after creating the
     * stream by using pa_stream_get_sample_spec(). Also, if you
     * specified manual buffer metrics it is recommended to update
     * them with pa_stream_set_buffer_attr() to compensate for the
     * changed frame sizes. Only supported when the server is at least
     * PA 0.9.8. It is ignored on older servers.
     *
     * When creating streams with pa_stream_new_extended(), this flag has no
     * effect. If you specify a format with PCM encoding, and you want the
     * server to choose the sample format, then you should leave the sample
     * format unspecified in the pa_format_info object. This also means that
     * you can't use pa_format_info_from_sample_spec(), because that function
     * always sets the sample format.
     *
     * \since 0.9.8 */

    PA_STREAM_FIX_RATE = 0x0080U,
    /**< Use the sample rate of the sink, and possibly ignore the rate
     * the sample spec contains. Usage similar to
     * PA_STREAM_FIX_FORMAT. Only supported when the server is at least
     * PA 0.9.8. It is ignored on older servers.
     *
     * When creating streams with pa_stream_new_extended(), this flag has no
     * effect. If you specify a format with PCM encoding, and you want the
     * server to choose the sample rate, then you should leave the rate
     * unspecified in the pa_format_info object. This also means that you can't
     * use pa_format_info_from_sample_spec(), because that function always sets
     * the sample rate.
     *
     * \since 0.9.8 */

    PA_STREAM_FIX_CHANNELS = 0x0100,
    /**< Use the number of channels and the channel map of the sink,
     * and possibly ignore the number of channels and the map the
     * sample spec and the passed channel map contains. Usage similar
     * to PA_STREAM_FIX_FORMAT. Only supported when the server is at
     * least PA 0.9.8. It is ignored on older servers.
     *
     * When creating streams with pa_stream_new_extended(), this flag has no
     * effect. If you specify a format with PCM encoding, and you want the
     * server to choose the channel count and/or channel map, then you should
     * leave the channels and/or the channel map unspecified in the
     * pa_format_info object. This also means that you can't use
     * pa_format_info_from_sample_spec(), because that function always sets
     * the channel count (but if you only want to leave the channel map
     * unspecified, then pa_format_info_from_sample_spec() works, because it
     * accepts a NULL channel map).
     *
     * \since 0.9.8 */

    PA_STREAM_DONT_MOVE = 0x0200U,
    /**< Don't allow moving of this stream to another
     * sink/device. Useful if you use any of the PA_STREAM_FIX_ flags
     * and want to make sure that resampling never takes place --
     * which might happen if the stream is moved to another
     * sink/source with a different sample spec/channel map. Only
     * supported when the server is at least PA 0.9.8. It is ignored
     * on older servers. \since 0.9.8 */

    PA_STREAM_VARIABLE_RATE = 0x0400U,
    /**< Allow dynamic changing of the sampling rate during playback
     * with pa_stream_update_sample_rate(). Only supported when the
     * server is at least PA 0.9.8. It is ignored on older
     * servers. \since 0.9.8 */

    PA_STREAM_PEAK_DETECT = 0x0800U,
    /**< Find peaks instead of resampling. \since 0.9.11 */

    PA_STREAM_START_MUTED = 0x1000U,
    /**< Create in muted state. If neither PA_STREAM_START_UNMUTED nor
     * PA_STREAM_START_MUTED it is left to the server to decide
     * whether to create the stream in muted or in unmuted
     * state. \since 0.9.11 */

    PA_STREAM_ADJUST_LATENCY = 0x2000U,
    /**< Try to adjust the latency of the sink/source based on the
     * requested buffer metrics and adjust buffer metrics
     * accordingly. Also see pa_buffer_attr. This option may not be
     * specified at the same time as PA_STREAM_EARLY_REQUESTS. \since
     * 0.9.11 */

    PA_STREAM_EARLY_REQUESTS = 0x4000U,
    /**< Enable compatibility mode for legacy clients that rely on a
     * "classic" hardware device fragment-style playback model. If
     * this option is set, the minreq value of the buffer metrics gets
     * a new meaning: instead of just specifying that no requests
     * asking for less new data than this value will be made to the
     * client it will also guarantee that requests are generated as
     * early as this limit is reached. This flag should only be set in
     * very few situations where compatibility with a fragment-based
     * playback model needs to be kept and the client applications
     * cannot deal with data requests that are delayed to the latest
     * moment possible. (Usually these are programs that use usleep()
     * or a similar call in their playback loops instead of sleeping
     * on the device itself.) Also see pa_buffer_attr. This option may
     * not be specified at the same time as
     * PA_STREAM_ADJUST_LATENCY. \since 0.9.12 */

    PA_STREAM_DONT_INHIBIT_AUTO_SUSPEND = 0x8000U,
    /**< If set this stream won't be taken into account when it is
     * checked whether the device this stream is connected to should
     * auto-suspend. \since 0.9.15 */

    PA_STREAM_START_UNMUTED = 0x10000U,
    /**< Create in unmuted state. If neither PA_STREAM_START_UNMUTED
     * nor PA_STREAM_START_MUTED it is left to the server to decide
     * whether to create the stream in muted or in unmuted
     * state. \since 0.9.15 */

    PA_STREAM_FAIL_ON_SUSPEND = 0x20000U,
    /**< If the sink/source this stream is connected to is suspended
     * during the creation of this stream, cause it to fail. If the
     * sink/source is being suspended during creation of this stream,
     * make sure this stream is terminated. \since 0.9.15 */

    PA_STREAM_RELATIVE_VOLUME = 0x40000U,
    /**< If a volume is passed when this stream is created, consider
     * it relative to the sink's current volume, never as absolute
     * device volume. If this is not specified the volume will be
     * consider absolute when the sink is in flat volume mode,
     * relative otherwise. \since 0.9.20 */

    PA_STREAM_PASSTHROUGH = 0x80000U
    /**< Used to tag content that will be rendered by passthrough sinks.
     * The data will be left as is and not reformatted, resampled.
     * \since 1.0 */

} pa_stream_flags_t;

/** \cond fulldocs */

/* English is an evil language */
#define PA_STREAM_NOT_MONOTONOUS PA_STREAM_NOT_MONOTONIC

/* Allow clients to check with #ifdef for those flags */
#define PA_STREAM_START_CORKED PA_STREAM_START_CORKED
#define PA_STREAM_INTERPOLATE_TIMING PA_STREAM_INTERPOLATE_TIMING
#define PA_STREAM_NOT_MONOTONIC PA_STREAM_NOT_MONOTONIC
#define PA_STREAM_AUTO_TIMING_UPDATE PA_STREAM_AUTO_TIMING_UPDATE
#define PA_STREAM_NO_REMAP_CHANNELS PA_STREAM_NO_REMAP_CHANNELS
#define PA_STREAM_NO_REMIX_CHANNELS PA_STREAM_NO_REMIX_CHANNELS
#define PA_STREAM_FIX_FORMAT PA_STREAM_FIX_FORMAT
#define PA_STREAM_FIX_RATE PA_STREAM_FIX_RATE
#define PA_STREAM_FIX_CHANNELS PA_STREAM_FIX_CHANNELS
#define PA_STREAM_DONT_MOVE PA_STREAM_DONT_MOVE
#define PA_STREAM_VARIABLE_RATE PA_STREAM_VARIABLE_RATE
#define PA_STREAM_PEAK_DETECT PA_STREAM_PEAK_DETECT
#define PA_STREAM_START_MUTED PA_STREAM_START_MUTED
#define PA_STREAM_ADJUST_LATENCY PA_STREAM_ADJUST_LATENCY
#define PA_STREAM_EARLY_REQUESTS PA_STREAM_EARLY_REQUESTS
#define PA_STREAM_DONT_INHIBIT_AUTO_SUSPEND PA_STREAM_DONT_INHIBIT_AUTO_SUSPEND
#define PA_STREAM_START_UNMUTED PA_STREAM_START_UNMUTED
#define PA_STREAM_FAIL_ON_SUSPEND PA_STREAM_FAIL_ON_SUSPEND
#define PA_STREAM_RELATIVE_VOLUME PA_STREAM_RELATIVE_VOLUME
#define PA_STREAM_PASSTHROUGH PA_STREAM_PASSTHROUGH

/** \endcond */

/** Playback and record buffer metrics */
typedef struct pa_buffer_attr {
    uint32_t maxlength;
    /**< Maximum length of the buffer in bytes. Setting this to (uint32_t) -1
     * will initialize this to the maximum value supported by server,
     * which is recommended.
     *
     * In strict low-latency playback scenarios you might want to set this to
     * a lower value, likely together with the PA_STREAM_ADJUST_LATENCY flag.
     * If you do so, you ensure that the latency doesn't grow beyond what is
     * acceptable for the use case, at the cost of getting more underruns if
     * the latency is lower than what the server can reliably handle. */

    uint32_t tlength;
    /**< Playback only: target length of the buffer. The server tries
     * to assure that at least tlength bytes are always available in
     * the per-stream server-side playback buffer. It is recommended
     * to set this to (uint32_t) -1, which will initialize this to a
     * value that is deemed sensible by the server. However, this
     * value will default to something like 2s, i.e. for applications
     * that have specific latency requirements this value should be
     * set to the maximum latency that the application can deal
     * with. When PA_STREAM_ADJUST_LATENCY is not set this value will
     * influence only the per-stream playback buffer size. When
     * PA_STREAM_ADJUST_LATENCY is set the overall latency of the sink
     * plus the playback buffer size is configured to this value. Set
     * PA_STREAM_ADJUST_LATENCY if you are interested in adjusting the
     * overall latency. Don't set it if you are interested in
     * configuring the server-side per-stream playback buffer
     * size. */

    uint32_t prebuf;
    /**< Playback only: pre-buffering. The server does not start with
     * playback before at least prebuf bytes are available in the
     * buffer. It is recommended to set this to (uint32_t) -1, which
     * will initialize this to the same value as tlength, whatever
     * that may be. Initialize to 0 to enable manual start/stop
     * control of the stream. This means that playback will not stop
     * on underrun and playback will not start automatically. Instead
     * pa_stream_cork() needs to be called explicitly. If you set
     * this value to 0 you should also set PA_STREAM_START_CORKED. */

    uint32_t minreq;
    /**< Playback only: minimum request. The server does not request
     * less than minreq bytes from the client, instead waits until the
     * buffer is free enough to request more bytes at once. It is
     * recommended to set this to (uint32_t) -1, which will initialize
     * this to a value that is deemed sensible by the server. This
     * should be set to a value that gives PulseAudio enough time to
     * move the data from the per-stream playback buffer into the
     * hardware playback buffer. */

    uint32_t fragsize;
    /**< Recording only: fragment size. The server sends data in
     * blocks of fragsize bytes size. Large values diminish
     * interactivity with other operations on the connection context
     * but decrease control overhead. It is recommended to set this to
     * (uint32_t) -1, which will initialize this to a value that is
     * deemed sensible by the server. However, this value will default
     * to something like 2s, i.e. for applications that have specific
     * latency requirements this value should be set to the maximum
     * latency that the application can deal with. If
     * PA_STREAM_ADJUST_LATENCY is set the overall source latency will
     * be adjusted according to this value. If it is not set the
     * source latency is left unmodified. */

} pa_buffer_attr;

/** Error values as used by pa_context_errno(). Use pa_strerror() to convert these values to human readable strings */
typedef enum pa_error_code {
    PA_OK = 0,                     /**< No error */
    PA_ERR_ACCESS,                 /**< Access failure */
    PA_ERR_COMMAND,                /**< Unknown command */
    PA_ERR_INVALID,                /**< Invalid argument */
    PA_ERR_EXIST,                  /**< Entity exists */
    PA_ERR_NOENTITY,               /**< No such entity */
    PA_ERR_CONNECTIONREFUSED,      /**< Connection refused */
    PA_ERR_PROTOCOL,               /**< Protocol error */
    PA_ERR_TIMEOUT,                /**< Timeout */
    PA_ERR_AUTHKEY,                /**< No authentication key */
    PA_ERR_INTERNAL,               /**< Internal error */
    PA_ERR_CONNECTIONTERMINATED,   /**< Connection terminated */
    PA_ERR_KILLED,                 /**< Entity killed */
    PA_ERR_INVALIDSERVER,          /**< Invalid server */
    PA_ERR_MODINITFAILED,          /**< Module initialization failed */
    PA_ERR_BADSTATE,               /**< Bad state */
    PA_ERR_NODATA,                 /**< No data */
    PA_ERR_VERSION,                /**< Incompatible protocol version */
    PA_ERR_TOOLARGE,               /**< Data too large */
    PA_ERR_NOTSUPPORTED,           /**< Operation not supported \since 0.9.5 */
    PA_ERR_UNKNOWN,                /**< The error code was unknown to the client */
    PA_ERR_NOEXTENSION,            /**< Extension does not exist. \since 0.9.12 */
    PA_ERR_OBSOLETE,               /**< Obsolete functionality. \since 0.9.15 */
    PA_ERR_NOTIMPLEMENTED,         /**< Missing implementation. \since 0.9.15 */
    PA_ERR_FORKED,                 /**< The caller forked without calling execve() and tried to reuse the context. \since 0.9.15 */
    PA_ERR_IO,                     /**< An IO error happened. \since 0.9.16 */
    PA_ERR_BUSY,                   /**< Device or resource busy. \since 0.9.17 */
    PA_ERR_MAX                     /**< Not really an error but the first invalid error code */
} pa_error_code_t;

/** \cond fulldocs */
#define PA_OK PA_OK
#define PA_ERR_ACCESS PA_ERR_ACCESS
#define PA_ERR_COMMAND PA_ERR_COMMAND
#define PA_ERR_INVALID PA_ERR_INVALID
#define PA_ERR_EXIST PA_ERR_EXIST
#define PA_ERR_NOENTITY PA_ERR_NOENTITY
#define PA_ERR_CONNECTIONREFUSED PA_ERR_CONNECTIONREFUSED
#define PA_ERR_PROTOCOL PA_ERR_PROTOCOL
#define PA_ERR_TIMEOUT PA_ERR_TIMEOUT
#define PA_ERR_AUTHKEY PA_ERR_AUTHKEY
#define PA_ERR_INTERNAL PA_ERR_INTERNAL
#define PA_ERR_CONNECTIONTERMINATED PA_ERR_CONNECTIONTERMINATED
#define PA_ERR_KILLED PA_ERR_KILLED
#define PA_ERR_INVALIDSERVER PA_ERR_INVALIDSERVER
#define PA_ERR_MODINITFAILED PA_ERR_MODINITFAILED
#define PA_ERR_BADSTATE PA_ERR_BADSTATE
#define PA_ERR_NODATA PA_ERR_NODATA
#define PA_ERR_VERSION PA_ERR_VERSION
#define PA_ERR_TOOLARGE PA_ERR_TOOLARGE
#define PA_ERR_NOTSUPPORTED PA_ERR_NOTSUPPORTED
#define PA_ERR_UNKNOWN PA_ERR_UNKNOWN
#define PA_ERR_NOEXTENSION PA_ERR_NOEXTENSION
#define PA_ERR_OBSOLETE PA_ERR_OBSOLETE
#define PA_ERR_NOTIMPLEMENTED PA_ERR_NOTIMPLEMENTED
#define PA_ERR_FORKED PA_ERR_FORKED
#define PA_ERR_MAX PA_ERR_MAX
/** \endcond */

/** Subscription event mask, as used by pa_context_subscribe() */
typedef enum pa_subscription_mask {
    PA_SUBSCRIPTION_MASK_NULL = 0x0000U,
    /**< No events */

    PA_SUBSCRIPTION_MASK_SINK = 0x0001U,
    /**< Sink events */

    PA_SUBSCRIPTION_MASK_SOURCE = 0x0002U,
    /**< Source events */

    PA_SUBSCRIPTION_MASK_SINK_INPUT = 0x0004U,
    /**< Sink input events */

    PA_SUBSCRIPTION_MASK_SOURCE_OUTPUT = 0x0008U,
    /**< Source output events */

    PA_SUBSCRIPTION_MASK_MODULE = 0x0010U,
    /**< Module events */

    PA_SUBSCRIPTION_MASK_CLIENT = 0x0020U,
    /**< Client events */

    PA_SUBSCRIPTION_MASK_SAMPLE_CACHE = 0x0040U,
    /**< Sample cache events */

    PA_SUBSCRIPTION_MASK_SERVER = 0x0080U,
    /**< Other global server changes. */

/** \cond fulldocs */
    PA_SUBSCRIPTION_MASK_AUTOLOAD = 0x0100U,
    /**< \deprecated Autoload table events. */
/** \endcond */

    PA_SUBSCRIPTION_MASK_CARD = 0x0200U,
    /**< Card events. \since 0.9.15 */

    PA_SUBSCRIPTION_MASK_ALL = 0x02ffU
    /**< Catch all events */
} pa_subscription_mask_t;

/** Subscription event types, as used by pa_context_subscribe() */
typedef enum pa_subscription_event_type {
    PA_SUBSCRIPTION_EVENT_SINK = 0x0000U,
    /**< Event type: Sink */

    PA_SUBSCRIPTION_EVENT_SOURCE = 0x0001U,
    /**< Event type: Source */

    PA_SUBSCRIPTION_EVENT_SINK_INPUT = 0x0002U,
    /**< Event type: Sink input */

    PA_SUBSCRIPTION_EVENT_SOURCE_OUTPUT = 0x0003U,
    /**< Event type: Source output */

    PA_SUBSCRIPTION_EVENT_MODULE = 0x0004U,
    /**< Event type: Module */

    PA_SUBSCRIPTION_EVENT_CLIENT = 0x0005U,
    /**< Event type: Client */

    PA_SUBSCRIPTION_EVENT_SAMPLE_CACHE = 0x0006U,
    /**< Event type: Sample cache item */

    PA_SUBSCRIPTION_EVENT_SERVER = 0x0007U,
    /**< Event type: Global server change, only occurring with PA_SUBSCRIPTION_EVENT_CHANGE. */

/** \cond fulldocs */
    PA_SUBSCRIPTION_EVENT_AUTOLOAD = 0x0008U,
    /**< \deprecated Event type: Autoload table changes. */
/** \endcond */

    PA_SUBSCRIPTION_EVENT_CARD = 0x0009U,
    /**< Event type: Card \since 0.9.15 */

    PA_SUBSCRIPTION_EVENT_FACILITY_MASK = 0x000FU,
    /**< A mask to extract the event type from an event value */

    PA_SUBSCRIPTION_EVENT_NEW = 0x0000U,
    /**< A new object was created */

    PA_SUBSCRIPTION_EVENT_CHANGE = 0x0010U,
    /**< A property of the object was modified */

    PA_SUBSCRIPTION_EVENT_REMOVE = 0x0020U,
    /**< An object was removed */

    PA_SUBSCRIPTION_EVENT_TYPE_MASK = 0x0030U
    /**< A mask to extract the event operation from an event value */

} pa_subscription_event_type_t;

/** Return one if an event type t matches an event mask bitfield */
#define pa_subscription_match_flags(m, t) (!!((m) & (1 << ((t) & PA_SUBSCRIPTION_EVENT_FACILITY_MASK))))

/** \cond fulldocs */
#define PA_SUBSCRIPTION_MASK_NULL PA_SUBSCRIPTION_MASK_NULL
#define PA_SUBSCRIPTION_MASK_SINK PA_SUBSCRIPTION_MASK_SINK
#define PA_SUBSCRIPTION_MASK_SOURCE PA_SUBSCRIPTION_MASK_SOURCE
#define PA_SUBSCRIPTION_MASK_SINK_INPUT PA_SUBSCRIPTION_MASK_SINK_INPUT
#define PA_SUBSCRIPTION_MASK_SOURCE_OUTPUT PA_SUBSCRIPTION_MASK_SOURCE_OUTPUT
#define PA_SUBSCRIPTION_MASK_MODULE PA_SUBSCRIPTION_MASK_MODULE
#define PA_SUBSCRIPTION_MASK_CLIENT PA_SUBSCRIPTION_MASK_CLIENT
#define PA_SUBSCRIPTION_MASK_SAMPLE_CACHE PA_SUBSCRIPTION_MASK_SAMPLE_CACHE
#define PA_SUBSCRIPTION_MASK_SERVER PA_SUBSCRIPTION_MASK_SERVER
#define PA_SUBSCRIPTION_MASK_AUTOLOAD PA_SUBSCRIPTION_MASK_AUTOLOAD
#define PA_SUBSCRIPTION_MASK_CARD PA_SUBSCRIPTION_MASK_CARD
#define PA_SUBSCRIPTION_MASK_ALL PA_SUBSCRIPTION_MASK_ALL
#define PA_SUBSCRIPTION_EVENT_SINK PA_SUBSCRIPTION_EVENT_SINK
#define PA_SUBSCRIPTION_EVENT_SOURCE PA_SUBSCRIPTION_EVENT_SOURCE
#define PA_SUBSCRIPTION_EVENT_SINK_INPUT PA_SUBSCRIPTION_EVENT_SINK_INPUT
#define PA_SUBSCRIPTION_EVENT_SOURCE_OUTPUT PA_SUBSCRIPTION_EVENT_SOURCE_OUTPUT
#define PA_SUBSCRIPTION_EVENT_MODULE PA_SUBSCRIPTION_EVENT_MODULE
#define PA_SUBSCRIPTION_EVENT_CLIENT PA_SUBSCRIPTION_EVENT_CLIENT
#define PA_SUBSCRIPTION_EVENT_SAMPLE_CACHE PA_SUBSCRIPTION_EVENT_SAMPLE_CACHE
#define PA_SUBSCRIPTION_EVENT_SERVER PA_SUBSCRIPTION_EVENT_SERVER
#define PA_SUBSCRIPTION_EVENT_AUTOLOAD PA_SUBSCRIPTION_EVENT_AUTOLOAD
#define PA_SUBSCRIPTION_EVENT_CARD PA_SUBSCRIPTION_EVENT_CARD
#define PA_SUBSCRIPTION_EVENT_FACILITY_MASK PA_SUBSCRIPTION_EVENT_FACILITY_MASK
#define PA_SUBSCRIPTION_EVENT_NEW PA_SUBSCRIPTION_EVENT_NEW
#define PA_SUBSCRIPTION_EVENT_CHANGE PA_SUBSCRIPTION_EVENT_CHANGE
#define PA_SUBSCRIPTION_EVENT_REMOVE PA_SUBSCRIPTION_EVENT_REMOVE
#define PA_SUBSCRIPTION_EVENT_TYPE_MASK PA_SUBSCRIPTION_EVENT_TYPE_MASK
/** \endcond */

/** A structure for all kinds of timing information of a stream. See
 * pa_stream_update_timing_info() and pa_stream_get_timing_info(). The
 * total output latency a sample that is written with
 * pa_stream_write() takes to be played may be estimated by
 * sink_usec+buffer_usec+transport_usec. (where buffer_usec is defined
 * as pa_bytes_to_usec(write_index-read_index)) The output buffer
 * which buffer_usec relates to may be manipulated freely (with
 * pa_stream_write()'s seek argument, pa_stream_flush() and friends),
 * the buffers sink_usec and source_usec relate to are first-in
 * first-out (FIFO) buffers which cannot be flushed or manipulated in
 * any way. The total input latency a sample that is recorded takes to
 * be delivered to the application is:
 * source_usec+buffer_usec+transport_usec-sink_usec. (Take care of
 * sign issues!) When connected to a monitor source sink_usec contains
 * the latency of the owning sink. The two latency estimations
 * described here are implemented in pa_stream_get_latency(). Please
 * note that this structure can be extended as part of evolutionary
 * API updates at any time in any new release.*/
typedef struct pa_timing_info {
    struct timeval timestamp;
    /**< The time when this timing info structure was current */

    int synchronized_clocks;
    /**< Non-zero if the local and the remote machine have
     * synchronized clocks. If synchronized clocks are detected
     * transport_usec becomes much more reliable. However, the code
     * that detects synchronized clocks is very limited and unreliable
     * itself. */

    pa_usec_t sink_usec;
    /**< Time in usecs a sample takes to be played on the sink. For
     * playback streams and record streams connected to a monitor
     * source. */

    pa_usec_t source_usec;
    /**< Time in usecs a sample takes from being recorded to being
     * delivered to the application. Only for record streams. */

    pa_usec_t transport_usec;
    /**< Estimated time in usecs a sample takes to be transferred
     * to/from the daemon. For both playback and record streams. */

    int playing;
    /**< Non-zero when the stream is currently not underrun and data
     * is being passed on to the device. Only for playback
     * streams. This field does not say whether the data is actually
     * already being played. To determine this check whether
     * since_underrun (converted to usec) is larger than sink_usec.*/

    int write_index_corrupt;
    /**< Non-zero if write_index is not up-to-date because a local
     * write command that corrupted it has been issued in the time
     * since this latency info was current . Only write commands with
     * SEEK_RELATIVE_ON_READ and SEEK_RELATIVE_END can corrupt
     * write_index. */

    int64_t write_index;
    /**< Current write index into the playback buffer in bytes. Think
     * twice before using this for seeking purposes: it might be out
     * of date a the time you want to use it. Consider using
     * PA_SEEK_RELATIVE instead. */

    int read_index_corrupt;
    /**< Non-zero if read_index is not up-to-date because a local
     * pause or flush request that corrupted it has been issued in the
     * time since this latency info was current. */

    int64_t read_index;
    /**< Current read index into the playback buffer in bytes. Think
     * twice before using this for seeking purposes: it might be out
     * of date a the time you want to use it. Consider using
     * PA_SEEK_RELATIVE_ON_READ instead. */

    pa_usec_t configured_sink_usec;
    /**< The configured latency for the sink. \since 0.9.11 */

    pa_usec_t configured_source_usec;
    /**< The configured latency for the source. \since 0.9.11 */

    int64_t since_underrun;
    /**< Bytes that were handed to the sink since the last underrun
     * happened, or since playback started again after the last
     * underrun. playing will tell you which case it is. \since
     * 0.9.11 */

} pa_timing_info;

/** A structure for the spawn api. This may be used to integrate auto
 * spawned daemons into your application. For more information see
 * pa_context_connect(). When spawning a new child process the
 * waitpid() is used on the child's PID. The spawn routine will not
 * block or ignore SIGCHLD signals, since this cannot be done in a
 * thread compatible way. You might have to do this in
 * prefork/postfork. */
typedef struct pa_spawn_api {
    void (*prefork)(void);
    /**< Is called just before the fork in the parent process. May be
     * NULL. */

    void (*postfork)(void);
    /**< Is called immediately after the fork in the parent
     * process. May be NULL.*/

    void (*atfork)(void);
    /**< Is called immediately after the fork in the child
     * process. May be NULL. It is not safe to close all file
     * descriptors in this function unconditionally, since a UNIX
     * socket (created using socketpair()) is passed to the new
     * process. */
} pa_spawn_api;

/** Seek type for pa_stream_write(). */
typedef enum pa_seek_mode {
    PA_SEEK_RELATIVE = 0,
    /**< Seek relatively to the write index */

    PA_SEEK_ABSOLUTE = 1,
    /**< Seek relatively to the start of the buffer queue */

    PA_SEEK_RELATIVE_ON_READ = 2,
    /**< Seek relatively to the read index.  */

    PA_SEEK_RELATIVE_END = 3
    /**< Seek relatively to the current end of the buffer queue. */
} pa_seek_mode_t;

/** \cond fulldocs */
#define PA_SEEK_RELATIVE PA_SEEK_RELATIVE
#define PA_SEEK_ABSOLUTE PA_SEEK_ABSOLUTE
#define PA_SEEK_RELATIVE_ON_READ PA_SEEK_RELATIVE_ON_READ
#define PA_SEEK_RELATIVE_END PA_SEEK_RELATIVE_END
/** \endcond */

/** Special sink flags. */
typedef enum pa_sink_flags {
    PA_SINK_NOFLAGS = 0x0000U,
    /**< Flag to pass when no specific options are needed (used to avoid casting)  \since 0.9.19 */

    PA_SINK_HW_VOLUME_CTRL = 0x0001U,
    /**< Supports hardware volume control. This is a dynamic flag and may
     * change at runtime after the sink has initialized */

    PA_SINK_LATENCY = 0x0002U,
    /**< Supports latency querying */

    PA_SINK_HARDWARE = 0x0004U,
    /**< Is a hardware sink of some kind, in contrast to
     * "virtual"/software sinks \since 0.9.3 */

    PA_SINK_NETWORK = 0x0008U,
    /**< Is a networked sink of some kind. \since 0.9.7 */

    PA_SINK_HW_MUTE_CTRL = 0x0010U,
    /**< Supports hardware mute control. This is a dynamic flag and may
     * change at runtime after the sink has initialized \since 0.9.11 */

    PA_SINK_DECIBEL_VOLUME = 0x0020U,
    /**< Volume can be translated to dB with pa_sw_volume_to_dB(). This is a
     * dynamic flag and may change at runtime after the sink has initialized
     * \since 0.9.11 */

    PA_SINK_FLAT_VOLUME = 0x0040U,
    /**< This sink is in flat volume mode, i.e.\ always the maximum of
     * the volume of all connected inputs. \since 0.9.15 */

    PA_SINK_DYNAMIC_LATENCY = 0x0080U,
    /**< The latency can be adjusted dynamically depending on the
     * needs of the connected streams. \since 0.9.15 */

    PA_SINK_SET_FORMATS = 0x0100U,
    /**< The sink allows setting what formats are supported by the connected
     * hardware. The actual functionality to do this might be provided by an
     * extension. \since 1.0 */

#ifdef __INCLUDED_FROM_PULSE_AUDIO
/** \cond fulldocs */
    /* PRIVATE: Server-side values -- do not try to use these at client-side.
     * The server will filter out these flags anyway, so you should never see
     * these flags in sinks. */

    PA_SINK_SHARE_VOLUME_WITH_MASTER = 0x1000000U,
    /**< This sink shares the volume with the master sink (used by some filter
     * sinks). */

    PA_SINK_DEFERRED_VOLUME = 0x2000000U,
    /**< The HW volume changes are syncronized with SW volume. */
/** \endcond */
#endif

} pa_sink_flags_t;

/** \cond fulldocs */
#define PA_SINK_HW_VOLUME_CTRL PA_SINK_HW_VOLUME_CTRL
#define PA_SINK_LATENCY PA_SINK_LATENCY
#define PA_SINK_HARDWARE PA_SINK_HARDWARE
#define PA_SINK_NETWORK PA_SINK_NETWORK
#define PA_SINK_HW_MUTE_CTRL PA_SINK_HW_MUTE_CTRL
#define PA_SINK_DECIBEL_VOLUME PA_SINK_DECIBEL_VOLUME
#define PA_SINK_FLAT_VOLUME PA_SINK_FLAT_VOLUME
#define PA_SINK_DYNAMIC_LATENCY PA_SINK_DYNAMIC_LATENCY
#define PA_SINK_SET_FORMATS PA_SINK_SET_FORMATS
#ifdef __INCLUDED_FROM_PULSE_AUDIO
#define PA_SINK_CLIENT_FLAGS_MASK 0xFFFFFF
#endif

/** \endcond */

/** Sink state. \since 0.9.15 */
typedef enum pa_sink_state { /* enum serialized in u8 */
    PA_SINK_INVALID_STATE = -1,
    /**< This state is used when the server does not support sink state introspection \since 0.9.15 */

    PA_SINK_RUNNING = 0,
    /**< Running, sink is playing and used by at least one non-corked sink-input \since 0.9.15 */

    PA_SINK_IDLE = 1,
    /**< When idle, the sink is playing but there is no non-corked sink-input attached to it \since 0.9.15 */

    PA_SINK_SUSPENDED = 2,
    /**< When suspended, actual sink access can be closed, for instance \since 0.9.15 */

/** \cond fulldocs */
    /* PRIVATE: Server-side values -- DO NOT USE THIS ON THE CLIENT
     * SIDE! These values are *not* considered part of the official PA
     * API/ABI. If you use them your application might break when PA
     * is upgraded. Also, please note that these values are not useful
     * on the client side anyway. */

    PA_SINK_INIT = -2,
    /**< Initialization state */

    PA_SINK_UNLINKED = -3
    /**< The state when the sink is getting unregistered and removed from client access */
/** \endcond */

} pa_sink_state_t;

/** Returns non-zero if sink is playing: running or idle. \since 0.9.15 */
static inline int PA_SINK_IS_OPENED(pa_sink_state_t x) {
    return x == PA_SINK_RUNNING || x == PA_SINK_IDLE;
}

/** Returns non-zero if sink is running. \since 1.0 */
static inline int PA_SINK_IS_RUNNING(pa_sink_state_t x) {
    return x == PA_SINK_RUNNING;
}

/** \cond fulldocs */
#define PA_SINK_INVALID_STATE PA_SINK_INVALID_STATE
#define PA_SINK_RUNNING PA_SINK_RUNNING
#define PA_SINK_IDLE PA_SINK_IDLE
#define PA_SINK_SUSPENDED PA_SINK_SUSPENDED
#define PA_SINK_INIT PA_SINK_INIT
#define PA_SINK_UNLINKED PA_SINK_UNLINKED
#define PA_SINK_IS_OPENED PA_SINK_IS_OPENED
/** \endcond */

/** Special source flags.  */
typedef enum pa_source_flags {
    PA_SOURCE_NOFLAGS = 0x0000U,
    /**< Flag to pass when no specific options are needed (used to avoid casting)  \since 0.9.19 */

    PA_SOURCE_HW_VOLUME_CTRL = 0x0001U,
    /**< Supports hardware volume control. This is a dynamic flag and may
     * change at runtime after the source has initialized */

    PA_SOURCE_LATENCY = 0x0002U,
    /**< Supports latency querying */

    PA_SOURCE_HARDWARE = 0x0004U,
    /**< Is a hardware source of some kind, in contrast to
     * "virtual"/software source \since 0.9.3 */

    PA_SOURCE_NETWORK = 0x0008U,
    /**< Is a networked source of some kind. \since 0.9.7 */

    PA_SOURCE_HW_MUTE_CTRL = 0x0010U,
    /**< Supports hardware mute control. This is a dynamic flag and may
     * change at runtime after the source has initialized \since 0.9.11 */

    PA_SOURCE_DECIBEL_VOLUME = 0x0020U,
    /**< Volume can be translated to dB with pa_sw_volume_to_dB(). This is a
     * dynamic flag and may change at runtime after the source has initialized
     * \since 0.9.11 */

    PA_SOURCE_DYNAMIC_LATENCY = 0x0040U,
    /**< The latency can be adjusted dynamically depending on the
     * needs of the connected streams. \since 0.9.15 */

    PA_SOURCE_FLAT_VOLUME = 0x0080U,
    /**< This source is in flat volume mode, i.e.\ always the maximum of
     * the volume of all connected outputs. \since 1.0 */

#ifdef __INCLUDED_FROM_PULSE_AUDIO
/** \cond fulldocs */
    /* PRIVATE: Server-side values -- do not try to use these at client-side.
     * The server will filter out these flags anyway, so you should never see
     * these flags in sources. */

    PA_SOURCE_SHARE_VOLUME_WITH_MASTER = 0x1000000U,
    /**< This source shares the volume with the master source (used by some filter
     * sources). */

    PA_SOURCE_DEFERRED_VOLUME = 0x2000000U,
    /**< The HW volume changes are syncronized with SW volume. */
#endif
} pa_source_flags_t;

/** \cond fulldocs */
#define PA_SOURCE_HW_VOLUME_CTRL PA_SOURCE_HW_VOLUME_CTRL
#define PA_SOURCE_LATENCY PA_SOURCE_LATENCY
#define PA_SOURCE_HARDWARE PA_SOURCE_HARDWARE
#define PA_SOURCE_NETWORK PA_SOURCE_NETWORK
#define PA_SOURCE_HW_MUTE_CTRL PA_SOURCE_HW_MUTE_CTRL
#define PA_SOURCE_DECIBEL_VOLUME PA_SOURCE_DECIBEL_VOLUME
#define PA_SOURCE_DYNAMIC_LATENCY PA_SOURCE_DYNAMIC_LATENCY
#define PA_SOURCE_FLAT_VOLUME PA_SOURCE_FLAT_VOLUME
#ifdef __INCLUDED_FROM_PULSE_AUDIO
#define PA_SOURCE_CLIENT_FLAGS_MASK 0xFFFFFF
#endif

/** \endcond */

/** Source state. \since 0.9.15 */
typedef enum pa_source_state {
    PA_SOURCE_INVALID_STATE = -1,
    /**< This state is used when the server does not support source state introspection \since 0.9.15 */

    PA_SOURCE_RUNNING = 0,
    /**< Running, source is recording and used by at least one non-corked source-output \since 0.9.15 */

    PA_SOURCE_IDLE = 1,
    /**< When idle, the source is still recording but there is no non-corked source-output \since 0.9.15 */

    PA_SOURCE_SUSPENDED = 2,
    /**< When suspended, actual source access can be closed, for instance \since 0.9.15 */

/** \cond fulldocs */
    /* PRIVATE: Server-side values -- DO NOT USE THIS ON THE CLIENT
     * SIDE! These values are *not* considered part of the official PA
     * API/ABI. If you use them your application might break when PA
     * is upgraded. Also, please note that these values are not useful
     * on the client side anyway. */

    PA_SOURCE_INIT = -2,
    /**< Initialization state */

    PA_SOURCE_UNLINKED = -3
    /**< The state when the source is getting unregistered and removed from client access */
/** \endcond */

} pa_source_state_t;

/** Returns non-zero if source is recording: running or idle. \since 0.9.15 */
static inline int PA_SOURCE_IS_OPENED(pa_source_state_t x) {
    return x == PA_SOURCE_RUNNING || x == PA_SOURCE_IDLE;
}

/** Returns non-zero if source is running \since 1.0 */
static inline int PA_SOURCE_IS_RUNNING(pa_source_state_t x) {
    return x == PA_SOURCE_RUNNING;
}

/** \cond fulldocs */
#define PA_SOURCE_INVALID_STATE PA_SOURCE_INVALID_STATE
#define PA_SOURCE_RUNNING PA_SOURCE_RUNNING
#define PA_SOURCE_IDLE PA_SOURCE_IDLE
#define PA_SOURCE_SUSPENDED PA_SOURCE_SUSPENDED
#define PA_SOURCE_INIT PA_SOURCE_INIT
#define PA_SOURCE_UNLINKED PA_SOURCE_UNLINKED
#define PA_SOURCE_IS_OPENED PA_SOURCE_IS_OPENED
/** \endcond */

/** A generic free() like callback prototype */
typedef void (*pa_free_cb_t)(void *p);

/** A stream policy/meta event requesting that an application should
 * cork a specific stream. See pa_stream_event_cb_t for more
 * information. \since 0.9.15 */
#define PA_STREAM_EVENT_REQUEST_CORK "request-cork"

/** A stream policy/meta event requesting that an application should
 * cork a specific stream. See pa_stream_event_cb_t for more
 * information, \since 0.9.15 */
#define PA_STREAM_EVENT_REQUEST_UNCORK "request-uncork"

/** A stream event notifying that the stream is going to be
 * disconnected because the underlying sink changed and no longer
 * supports the format that was originally negotiated. Clients need
 * to connect a new stream to renegotiate a format and continue
 * playback. \since 1.0 */
#define PA_STREAM_EVENT_FORMAT_LOST "format-lost"

#ifndef __INCLUDED_FROM_PULSE_AUDIO
/** Port availability / jack detection status
 * \since 2.0 */
typedef enum pa_port_available {
    PA_PORT_AVAILABLE_UNKNOWN = 0, /**< This port does not support jack detection \since 2.0 */
    PA_PORT_AVAILABLE_NO = 1,      /**< This port is not available, likely because the jack is not plugged in. \since 2.0 */
    PA_PORT_AVAILABLE_YES = 2,     /**< This port is available, likely because the jack is plugged in. \since 2.0 */
} pa_port_available_t;

/** \cond fulldocs */
#define PA_PORT_AVAILABLE_UNKNOWN PA_PORT_AVAILABLE_UNKNOWN
#define PA_PORT_AVAILABLE_NO PA_PORT_AVAILABLE_NO
#define PA_PORT_AVAILABLE_YES PA_PORT_AVAILABLE_YES

/** \endcond */
#endif

PA_C_DECL_END

#endif
