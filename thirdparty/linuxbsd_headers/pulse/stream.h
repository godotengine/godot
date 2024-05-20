#ifndef foostreamhfoo
#define foostreamhfoo

/***
  This file is part of PulseAudio.

  Copyright 2004-2006 Lennart Poettering
  Copyright 2006 Pierre Ossman <ossman@cendio.se> for Cendio AB

  PulseAudio is free software; you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as published
  by the Free Software Foundation; either version 2.1 of the License,
  or (at your option) any later version.

  PulseAudio is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
  General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with PulseAudio; if not, see <http://www.gnu.org/licenses/>.
***/

#include <sys/types.h>

#include <pulse/sample.h>
#include <pulse/format.h>
#include <pulse/channelmap.h>
#include <pulse/volume.h>
#include <pulse/def.h>
#include <pulse/cdecl.h>
#include <pulse/operation.h>
#include <pulse/context.h>
#include <pulse/proplist.h>

/** \page streams Audio Streams
 *
 * \section overv_sec Overview
 *
 * Audio streams form the central functionality of the sound server. Data is
 * routed, converted and mixed from several sources before it is passed along
 * to a final output. Currently, there are three forms of audio streams:
 *
 * \li Playback streams - Data flows from the client to the server.
 * \li Record streams - Data flows from the server to the client.
 * \li Upload streams - Similar to playback streams, but the data is stored in
 *                      the sample cache. See \ref scache for more information
 *                      about controlling the sample cache.
 *
 * \section create_sec Creating
 *
 * To access a stream, a pa_stream object must be created using
 * pa_stream_new() or pa_stream_new_extended(). pa_stream_new() is for PCM
 * streams only, while pa_stream_new_extended() can be used for both PCM and
 * compressed audio streams. At this point the application must specify what
 * stream format(s) it supports. See \ref sample and \ref channelmap for more
 * information on the stream format parameters. FIXME: Those references only
 * talk about PCM parameters, we should also have an overview page for how the
 * pa_format_info based stream format configuration works. Bug filed:
 * https://bugs.freedesktop.org/show_bug.cgi?id=72265
 *
 * This first step will only create a client-side object, representing the
 * stream. To use the stream, a server-side object must be created and
 * associated with the local object. Depending on which type of stream is
 * desired, a different function is needed:
 *
 * \li Playback stream - pa_stream_connect_playback()
 * \li Record stream - pa_stream_connect_record()
 * \li Upload stream - pa_stream_connect_upload() (see \ref scache)
 *
 * Similar to how connections are done in contexts, connecting a stream will
 * not generate a pa_operation object. Also like contexts, the application
 * should register a state change callback, using
 * pa_stream_set_state_callback(), and wait for the stream to enter an active
 * state.
 *
 * Note: there is a user-controllable slider in mixer applications such as
 * pavucontrol corresponding to each of the created streams. Multiple
 * (especially identically named) volume sliders for the same application might
 * confuse the user. Also, the server supports only a limited number of
 * simultaneous streams. Because of this, it is not always appropriate to
 * create multiple streams in one application that needs to output multiple
 * sounds. The rough guideline is: if there is no use case that would require
 * separate user-initiated volume changes for each stream, perform the mixing
 * inside the application.
 *
 * \subsection bufattr_subsec Buffer Attributes
 *
 * Playback and record streams always have a server-side buffer as
 * part of the data flow.  The size of this buffer needs to be chosen
 * in a compromise between low latency and sensitivity for buffer
 * overflows/underruns.
 *
 * The buffer metrics may be controlled by the application. They are
 * described with a pa_buffer_attr structure which contains a number
 * of fields:
 *
 * \li maxlength - The absolute maximum number of bytes that can be
 *                 stored in the buffer. If this value is exceeded
 *                 then data will be lost. It is recommended to pass
 *                 (uint32_t) -1 here which will cause the server to
 *                 fill in the maximum possible value.
 *
 * \li tlength - The target fill level of the playback buffer. The
 *               server will only send requests for more data as long
 *               as the buffer has less than this number of bytes of
 *               data. If you pass (uint32_t) -1 (which is
 *               recommended) here the server will choose the longest
 *               target buffer fill level possible to minimize the
 *               number of necessary wakeups and maximize drop-out
 *               safety. This can exceed 2s of buffering. For
 *               low-latency applications or applications where
 *               latency matters you should pass a proper value here.
 *
 * \li prebuf - Number of bytes that need to be in the buffer before
 *              playback will commence. Start of playback can be
 *              forced using pa_stream_trigger() even though the
 *              prebuffer size hasn't been reached. If a buffer
 *              underrun occurs, this prebuffering will be again
 *              enabled. If the playback shall never stop in case of a
 *              buffer underrun, this value should be set to 0. In
 *              that case the read index of the output buffer
 *              overtakes the write index, and hence the fill level of
 *              the buffer is negative. If you pass (uint32_t) -1 here
 *              (which is recommended) the server will choose the same
 *              value as tlength here.
 *
 * \li minreq - Minimum number of free bytes in the playback
 *              buffer before the server will request more data. It is
 *              recommended to fill in (uint32_t) -1 here. This value
 *              influences how much time the sound server has to move
 *              data from the per-stream server-side playback buffer
 *              to the hardware playback buffer.
 *
 * \li fragsize - Maximum number of bytes that the server will push in
 *                one chunk for record streams. If you pass (uint32_t)
 *                -1 (which is recommended) here, the server will
 *                choose the longest fragment setting possible to
 *                minimize the number of necessary wakeups and
 *                maximize drop-out safety. This can exceed 2s of
 *                buffering. For low-latency applications or
 *                applications where latency matters you should pass a
 *                proper value here.
 *
 * If PA_STREAM_ADJUST_LATENCY is set, then the tlength/fragsize
 * parameters will be interpreted slightly differently than described
 * above when passed to pa_stream_connect_record() and
 * pa_stream_connect_playback(): the overall latency that is comprised
 * of both the server side playback buffer length, the hardware
 * playback buffer length and additional latencies will be adjusted in
 * a way that it matches tlength resp. fragsize. Set
 * PA_STREAM_ADJUST_LATENCY if you want to control the overall
 * playback latency for your stream. Unset it if you want to control
 * only the latency induced by the server-side, rewritable playback
 * buffer. The server will try to fulfill the client's latency requests
 * as good as possible. However if the underlying hardware cannot
 * change the hardware buffer length or only in a limited range, the
 * actually resulting latency might be different from what the client
 * requested. Thus, for synchronization clients always need to check
 * the actual measured latency via pa_stream_get_latency() or a
 * similar call, and not make any assumptions about the latency
 * available. The function pa_stream_get_buffer_attr() will always
 * return the actual size of the server-side per-stream buffer in
 * tlength/fragsize, regardless whether PA_STREAM_ADJUST_LATENCY is
 * set or not.
 *
 * The server-side per-stream playback buffers are indexed by a write and a read
 * index. The application writes to the write index and the sound
 * device reads from the read index. The read index is increased
 * monotonically, while the write index may be freely controlled by
 * the application. Subtracting the read index from the write index
 * will give you the current fill level of the buffer. The read/write
 * indexes are 64bit values and measured in bytes, they will never
 * wrap. The current read/write index may be queried using
 * pa_stream_get_timing_info() (see below for more information). In
 * case of a buffer underrun the read index is equal or larger than
 * the write index. Unless the prebuf value is 0, PulseAudio will
 * temporarily pause playback in such a case, and wait until the
 * buffer is filled up to prebuf bytes again. If prebuf is 0, the
 * read index may be larger than the write index, in which case
 * silence is played. If the application writes data to indexes lower
 * than the read index, the data is immediately lost.
 *
 * \section transfer_sec Transferring Data
 *
 * Once the stream is up, data can start flowing between the client and the
 * server. Two different access models can be used to transfer the data:
 *
 * \li Asynchronous - The application register a callback using
 *                    pa_stream_set_write_callback() and
 *                    pa_stream_set_read_callback() to receive notifications
 *                    that data can either be written or read.
 * \li Polled - Query the library for available data/space using
 *              pa_stream_writable_size() and pa_stream_readable_size() and
 *              transfer data as needed. The sizes are stored locally, in the
 *              client end, so there is no delay when reading them.
 *
 * It is also possible to mix the two models freely.
 *
 * Once there is data/space available, it can be transferred using either
 * pa_stream_write() for playback, or pa_stream_peek() / pa_stream_drop() for
 * record. Make sure you do not overflow the playback buffers as data will be
 * dropped.
 *
 * \section bufctl_sec Buffer Control
 *
 * The transfer buffers can be controlled through a number of operations:
 *
 * \li pa_stream_cork() - Start or stop the playback or recording.
 * \li pa_stream_trigger() - Start playback immediately and do not wait for
 *                           the buffer to fill up to the set trigger level.
 * \li pa_stream_prebuf() - Reenable the playback trigger level.
 * \li pa_stream_drain() - Wait for the playback buffer to go empty. Will
 *                         return a pa_operation object that will indicate when
 *                         the buffer is completely drained.
 * \li pa_stream_flush() - Drop all data from the playback or record buffer. Do not
 *                         wait for it to finish playing.
 *
 * \section seek_modes Seeking in the Playback Buffer
 *
 * A client application may freely seek in the playback buffer. To
 * accomplish that the pa_stream_write() function takes a seek mode
 * and an offset argument. The seek mode is one of:
 *
 * \li PA_SEEK_RELATIVE - seek relative to the current write index
 * \li PA_SEEK_ABSOLUTE - seek relative to the beginning of the playback buffer, (i.e. the first that was ever played in the stream)
 * \li PA_SEEK_RELATIVE_ON_READ - seek relative to the current read index. Use this to write data to the output buffer that should be played as soon as possible
 * \li PA_SEEK_RELATIVE_END - seek relative to the last byte ever written.
 *
 * If an application just wants to append some data to the output
 * buffer, PA_SEEK_RELATIVE and an offset of 0 should be used.
 *
 * After a call to pa_stream_write() the write index will be left at
 * the position right after the last byte of the written data.
 *
 * \section latency_sec Latency
 *
 * A major problem with networked audio is the increased latency caused by
 * the network. To remedy this, PulseAudio supports an advanced system of
 * monitoring the current latency.
 *
 * To get the raw data needed to calculate latencies, call
 * pa_stream_get_timing_info(). This will give you a pa_timing_info
 * structure that contains everything that is known about the server
 * side buffer transport delays and the backend active in the
 * server. (Besides other things it contains the write and read index
 * values mentioned above.)
 *
 * This structure is updated every time a
 * pa_stream_update_timing_info() operation is executed. (i.e. before
 * the first call to this function the timing information structure is
 * not available!) Since it is a lot of work to keep this structure
 * up-to-date manually, PulseAudio can do that automatically for you:
 * if PA_STREAM_AUTO_TIMING_UPDATE is passed when connecting the
 * stream PulseAudio will automatically update the structure every
 * 100ms and every time a function is called that might invalidate the
 * previously known timing data (such as pa_stream_write() or
 * pa_stream_flush()). Please note however, that there always is a
 * short time window when the data in the timing information structure
 * is out-of-date. PulseAudio tries to mark these situations by
 * setting the write_index_corrupt and read_index_corrupt fields
 * accordingly.
 *
 * The raw timing data in the pa_timing_info structure is usually hard
 * to deal with. Therefore a simpler interface is available:
 * you can call pa_stream_get_time() or pa_stream_get_latency(). The
 * former will return the current playback time of the hardware since
 * the stream has been started. The latter returns the overall time a sample
 * that you write now takes to be played by the hardware. These two
 * functions base their calculations on the same data that is returned
 * by pa_stream_get_timing_info(). Hence the same rules for keeping
 * the timing data up-to-date apply here. In case the write or read
 * index is corrupted, these two functions will fail with
 * -PA_ERR_NODATA set.
 *
 * Since updating the timing info structure usually requires a full
 * network round trip and some applications monitor the timing very
 * often PulseAudio offers a timing interpolation system. If
 * PA_STREAM_INTERPOLATE_TIMING is passed when connecting the stream,
 * pa_stream_get_time() and pa_stream_get_latency() will try to
 * interpolate the current playback time/latency by estimating the
 * number of samples that have been played back by the hardware since
 * the last regular timing update. It is especially useful to combine
 * this option with PA_STREAM_AUTO_TIMING_UPDATE, which will enable
 * you to monitor the current playback time/latency very precisely and
 * very frequently without requiring a network round trip every time.
 *
 * \section flow_sec Overflow and underflow
 *
 * Even with the best precautions, buffers will sometime over - or
 * underflow.  To handle this gracefully, the application can be
 * notified when this happens. Callbacks are registered using
 * pa_stream_set_overflow_callback() and
 * pa_stream_set_underflow_callback().
 *
 * \section sync_streams Synchronizing Multiple Playback Streams
 *
 * PulseAudio allows applications to fully synchronize multiple
 * playback streams that are connected to the same output device. That
 * means the streams will always be played back sample-by-sample
 * synchronously. If stream operations like pa_stream_cork() are
 * issued on one of the synchronized streams, they are simultaneously
 * issued on the others.
 *
 * To synchronize a stream to another, just pass the "master" stream
 * as last argument to pa_stream_connect_playback(). To make sure that
 * the freshly created stream doesn't start playback right-away, make
 * sure to pass PA_STREAM_START_CORKED and -- after all streams have
 * been created -- uncork them all with a single call to
 * pa_stream_cork() for the master stream.
 *
 * To make sure that a particular stream doesn't stop to play when a
 * server side buffer underrun happens on it while the other
 * synchronized streams continue playing and hence deviate, you need to
 * pass a "prebuf" pa_buffer_attr of 0 when connecting it.
 *
 * \section disc_sec Disconnecting
 *
 * When a stream has served is purpose it must be disconnected with
 * pa_stream_disconnect(). If you only unreference it, then it will live on
 * and eat resources both locally and on the server until you disconnect the
 * context.
 *
 */

/** \file
 * Audio streams for input, output and sample upload
 *
 * See also \subpage streams
 */

PA_C_DECL_BEGIN

/** An opaque stream for playback or recording */
typedef struct pa_stream pa_stream;

/** A generic callback for operation completion */
typedef void (*pa_stream_success_cb_t) (pa_stream*s, int success, void *userdata);

/** A generic request callback */
typedef void (*pa_stream_request_cb_t)(pa_stream *p, size_t nbytes, void *userdata);

/** A generic notification callback */
typedef void (*pa_stream_notify_cb_t)(pa_stream *p, void *userdata);

/** A callback for asynchronous meta/policy event messages. Well known
 * event names are PA_STREAM_EVENT_REQUEST_CORK and
 * PA_STREAM_EVENT_REQUEST_UNCORK. The set of defined events can be
 * extended at any time. Also, server modules may introduce additional
 * message types so make sure that your callback function ignores messages
 * it doesn't know. \since 0.9.15 */
typedef void (*pa_stream_event_cb_t)(pa_stream *p, const char *name, pa_proplist *pl, void *userdata);

/** Create a new, unconnected stream with the specified name and
 * sample type. It is recommended to use pa_stream_new_with_proplist()
 * instead and specify some initial properties. */
pa_stream* pa_stream_new(
        pa_context *c                     /**< The context to create this stream in */,
        const char *name                  /**< A name for this stream */,
        const pa_sample_spec *ss          /**< The desired sample format */,
        const pa_channel_map *map         /**< The desired channel map, or NULL for default */);

/** Create a new, unconnected stream with the specified name and
 * sample type, and specify the initial stream property
 * list. \since 0.9.11 */
pa_stream* pa_stream_new_with_proplist(
        pa_context *c                     /**< The context to create this stream in */,
        const char *name                  /**< A name for this stream */,
        const pa_sample_spec *ss          /**< The desired sample format */,
        const pa_channel_map *map         /**< The desired channel map, or NULL for default */,
        pa_proplist *p                    /**< The initial property list */);

/** Create a new, unconnected stream with the specified name, the set of formats
 * this client can provide, and an initial list of properties. While
 * connecting, the server will select the most appropriate format which the
 * client must then provide. \since 1.0 */
pa_stream *pa_stream_new_extended(
        pa_context *c                     /**< The context to create this stream in */,
        const char *name                  /**< A name for this stream */,
        pa_format_info * const * formats  /**< The list of formats that can be provided */,
        unsigned int n_formats            /**< The number of formats being passed in */,
        pa_proplist *p                    /**< The initial property list */);

/** Decrease the reference counter by one. */
void pa_stream_unref(pa_stream *s);

/** Increase the reference counter by one. */
pa_stream *pa_stream_ref(pa_stream *s);

/** Return the current state of the stream. */
pa_stream_state_t pa_stream_get_state(pa_stream *p);

/** Return the context this stream is attached to. */
pa_context* pa_stream_get_context(pa_stream *p);

/** Return the sink input resp.\ source output index this stream is
 * identified in the server with. This is useful with the
 * introspection functions such as pa_context_get_sink_input_info()
 * or pa_context_get_source_output_info(). */
uint32_t pa_stream_get_index(pa_stream *s);

/** Return the index of the sink or source this stream is connected to
 * in the server. This is useful with the introspection
 * functions such as pa_context_get_sink_info_by_index() or
 * pa_context_get_source_info_by_index().
 *
 * Please note that streams may be moved between sinks/sources and thus
 * it is recommended to use pa_stream_set_moved_callback() to be notified
 * about this. This function will return with -PA_ERR_NOTSUPPORTED when the
 * server is older than 0.9.8. \since 0.9.8 */
uint32_t pa_stream_get_device_index(pa_stream *s);

/** Return the name of the sink or source this stream is connected to
 * in the server. This is useful with the introspection
 * functions such as pa_context_get_sink_info_by_name()
 * or pa_context_get_source_info_by_name().
 *
 * Please note that streams may be moved between sinks/sources and thus
 * it is recommended to use pa_stream_set_moved_callback() to be notified
 * about this. This function will return with -PA_ERR_NOTSUPPORTED when the
 * server is older than 0.9.8. \since 0.9.8 */
const char *pa_stream_get_device_name(pa_stream *s);

/** Return 1 if the sink or source this stream is connected to has
 * been suspended. This will return 0 if not, and a negative value on
 * error. This function will return with -PA_ERR_NOTSUPPORTED when the
 * server is older than 0.9.8. \since 0.9.8 */
int pa_stream_is_suspended(pa_stream *s);

/** Return 1 if the this stream has been corked. This will return 0 if
 * not, and a negative value on error. \since 0.9.11 */
int pa_stream_is_corked(pa_stream *s);

/** Connect the stream to a sink. It is strongly recommended to pass
 * NULL in both \a dev and \a volume and to set neither
 * PA_STREAM_START_MUTED nor PA_STREAM_START_UNMUTED -- unless these
 * options are directly dependent on user input or configuration.
 *
 * If you follow this rule then the sound server will have the full
 * flexibility to choose the device, volume and mute status
 * automatically, based on server-side policies, heuristics and stored
 * information from previous uses. Also the server may choose to
 * reconfigure audio devices to make other sinks/sources or
 * capabilities available to be able to accept the stream.
 *
 * Before 0.9.20 it was not defined whether the \a volume parameter was
 * interpreted relative to the sink's current volume or treated as
 * an absolute device volume. Since 0.9.20 it is an absolute volume when
 * the sink is in flat volume mode, and relative otherwise, thus
 * making sure the volume passed here has always the same semantics as
 * the volume passed to pa_context_set_sink_input_volume(). It is possible
 * to figure out whether flat volume mode is in effect for a given sink
 * by calling pa_context_get_sink_info_by_name().
 *
 * Since 5.0, it's possible to specify a single-channel volume even if the
 * stream has multiple channels. In that case the same volume is applied to all
 * channels. */
int pa_stream_connect_playback(
        pa_stream *s                  /**< The stream to connect to a sink */,
        const char *dev               /**< Name of the sink to connect to, or NULL for default */ ,
        const pa_buffer_attr *attr    /**< Buffering attributes, or NULL for default */,
        pa_stream_flags_t flags       /**< Additional flags, or 0 for default */,
        const pa_cvolume *volume      /**< Initial volume, or NULL for default */,
        pa_stream *sync_stream        /**< Synchronize this stream with the specified one, or NULL for a standalone stream */);

/** Connect the stream to a source. */
int pa_stream_connect_record(
        pa_stream *s                  /**< The stream to connect to a source */ ,
        const char *dev               /**< Name of the source to connect to, or NULL for default */,
        const pa_buffer_attr *attr    /**< Buffer attributes, or NULL for default */,
        pa_stream_flags_t flags       /**< Additional flags, or 0 for default */);

/** Disconnect a stream from a source/sink. */
int pa_stream_disconnect(pa_stream *s);

/** Prepare writing data to the server (for playback streams). This
 * function may be used to optimize the number of memory copies when
 * doing playback ("zero-copy"). It is recommended to call this
 * function before each call to pa_stream_write().
 *
 * Pass in the address to a pointer and an address of the number of
 * bytes you want to write. On return the two values will contain a
 * pointer where you can place the data to write and the maximum number
 * of bytes you can write. \a *nbytes can be smaller or have the same
 * value as you passed in. You need to be able to handle both cases.
 * Accessing memory beyond the returned \a *nbytes value is invalid.
 * Accessing the memory returned after the following pa_stream_write()
 * or pa_stream_cancel_write() is invalid.
 *
 * On invocation only \a *nbytes needs to be initialized, on return both
 * *data and *nbytes will be valid. If you place (size_t) -1 in *nbytes
 * on invocation the memory size will be chosen automatically (which is
 * recommended to do). After placing your data in the memory area
 * returned, call pa_stream_write() with \a data set to an address
 * within this memory area and an \a nbytes value that is smaller or
 * equal to what was returned by this function to actually execute the
 * write.
 *
 * An invocation of pa_stream_write() should follow "quickly" on
 * pa_stream_begin_write(). It is not recommended letting an unbounded
 * amount of time pass after calling pa_stream_begin_write() and
 * before calling pa_stream_write(). If you want to cancel a
 * previously called pa_stream_begin_write() without calling
 * pa_stream_write() use pa_stream_cancel_write(). Calling
 * pa_stream_begin_write() twice without calling pa_stream_write() or
 * pa_stream_cancel_write() in between will return exactly the same
 * \a data pointer and \a nbytes values. \since 0.9.16 */
int pa_stream_begin_write(
        pa_stream *p,
        void **data,
        size_t *nbytes);

/** Reverses the effect of pa_stream_begin_write() dropping all data
 * that has already been placed in the memory area returned by
 * pa_stream_begin_write(). Only valid to call if
 * pa_stream_begin_write() was called before and neither
 * pa_stream_cancel_write() nor pa_stream_write() have been called
 * yet. Accessing the memory previously returned by
 * pa_stream_begin_write() after this call is invalid. Any further
 * explicit freeing of the memory area is not necessary. \since
 * 0.9.16 */
int pa_stream_cancel_write(
        pa_stream *p);

/** Write some data to the server (for playback streams).
 * If \a free_cb is non-NULL this routine is called when all data has
 * been written out. An internal reference to the specified data is
 * kept, the data is not copied. If NULL, the data is copied into an
 * internal buffer.
 *
 * The client may freely seek around in the output buffer. For
 * most applications it is typical to pass 0 and PA_SEEK_RELATIVE
 * as values for the arguments \a offset and \a seek. After the write
 * call succeeded the write index will be at the position after where
 * this chunk of data has been written to.
 *
 * As an optimization for avoiding needless memory copies you may call
 * pa_stream_begin_write() before this call and then place your audio
 * data directly in the memory area returned by that call. Then, pass
 * a pointer to that memory area to pa_stream_write(). After the
 * invocation of pa_stream_write() the memory area may no longer be
 * accessed. Any further explicit freeing of the memory area is not
 * necessary. It is OK to write the memory area returned by
 * pa_stream_begin_write() only partially with this call, skipping
 * bytes both at the end and at the beginning of the reserved memory
 * area.*/
int pa_stream_write(
        pa_stream *p             /**< The stream to use */,
        const void *data         /**< The data to write */,
        size_t nbytes            /**< The length of the data to write in bytes, must be in multiples of the stream's sample spec frame size */,
        pa_free_cb_t free_cb     /**< A cleanup routine for the data or NULL to request an internal copy */,
        int64_t offset           /**< Offset for seeking, must be 0 for upload streams, must be in multiples of the stream's sample spec frame size */,
        pa_seek_mode_t seek      /**< Seek mode, must be PA_SEEK_RELATIVE for upload streams */);

/** Function does exactly the same as pa_stream_write() with the difference
 *  that free_cb_data is passed to free_cb instead of data. \since 6.0 */
int pa_stream_write_ext_free(
        pa_stream *p             /**< The stream to use */,
        const void *data         /**< The data to write */,
        size_t nbytes            /**< The length of the data to write in bytes */,
        pa_free_cb_t free_cb     /**< A cleanup routine for the data or NULL to request an internal copy */,
        void *free_cb_data       /**< Argument passed to free_cb function */,
        int64_t offset           /**< Offset for seeking, must be 0 for upload streams */,
        pa_seek_mode_t seek      /**< Seek mode, must be PA_SEEK_RELATIVE for upload streams */);

/** Read the next fragment from the buffer (for recording streams).
 * If there is data at the current read index, \a data will point to
 * the actual data and \a nbytes will contain the size of the data in
 * bytes (which can be less or more than a complete fragment).
 *
 * If there is no data at the current read index, it means that either
 * the buffer is empty or it contains a hole (that is, the write index
 * is ahead of the read index but there's no data where the read index
 * points at). If the buffer is empty, \a data will be NULL and
 * \a nbytes will be 0. If there is a hole, \a data will be NULL and
 * \a nbytes will contain the length of the hole.
 *
 * Use pa_stream_drop() to actually remove the data from the buffer
 * and move the read index forward. pa_stream_drop() should not be
 * called if the buffer is empty, but it should be called if there is
 * a hole. */
int pa_stream_peek(
        pa_stream *p                 /**< The stream to use */,
        const void **data            /**< Pointer to pointer that will point to data */,
        size_t *nbytes               /**< The length of the data read in bytes */);

/** Remove the current fragment on record streams. It is invalid to do this without first
 * calling pa_stream_peek(). */
int pa_stream_drop(pa_stream *p);

/** Return the number of bytes requested by the server that have not yet
 * been written.
 *
 * It is possible to write more than this amount, up to the stream's
 * buffer_attr.maxlength bytes. This is usually not desirable, though, as
 * it would increase stream latency to be higher than requested
 * (buffer_attr.tlength).
 */
size_t pa_stream_writable_size(pa_stream *p);

/** Return the number of bytes that may be read using pa_stream_peek(). */
size_t pa_stream_readable_size(pa_stream *p);

/** Drain a playback stream.  Use this for notification when the
 * playback buffer is empty after playing all the audio in the buffer.
 * Please note that only one drain operation per stream may be issued
 * at a time. */
pa_operation* pa_stream_drain(pa_stream *s, pa_stream_success_cb_t cb, void *userdata);

/** Request a timing info structure update for a stream. Use
 * pa_stream_get_timing_info() to get access to the raw timing data,
 * or pa_stream_get_time() or pa_stream_get_latency() to get cleaned
 * up values. */
pa_operation* pa_stream_update_timing_info(pa_stream *p, pa_stream_success_cb_t cb, void *userdata);

/** Set the callback function that is called whenever the state of the stream changes. */
void pa_stream_set_state_callback(pa_stream *s, pa_stream_notify_cb_t cb, void *userdata);

/** Set the callback function that is called when new data may be
 * written to the stream. */
void pa_stream_set_write_callback(pa_stream *p, pa_stream_request_cb_t cb, void *userdata);

/** Set the callback function that is called when new data is available from the stream. */
void pa_stream_set_read_callback(pa_stream *p, pa_stream_request_cb_t cb, void *userdata);

/** Set the callback function that is called when a buffer overflow happens. (Only for playback streams) */
void pa_stream_set_overflow_callback(pa_stream *p, pa_stream_notify_cb_t cb, void *userdata);

/** Return at what position the latest underflow occurred, or -1 if this information is not
 * known (e.g.\ if no underflow has occurred, or server is older than 1.0).
 * Can be used inside the underflow callback to get information about the current underflow.
 * (Only for playback streams) \since 1.0 */
int64_t pa_stream_get_underflow_index(pa_stream *p);

/** Set the callback function that is called when a buffer underflow happens. (Only for playback streams) */
void pa_stream_set_underflow_callback(pa_stream *p, pa_stream_notify_cb_t cb, void *userdata);

/** Set the callback function that is called when a the server starts
 * playback after an underrun or on initial startup. This only informs
 * that audio is flowing again, it is no indication that audio started
 * to reach the speakers already. (Only for playback streams) \since
 * 0.9.11 */
void pa_stream_set_started_callback(pa_stream *p, pa_stream_notify_cb_t cb, void *userdata);

/** Set the callback function that is called whenever a latency
 * information update happens. Useful on PA_STREAM_AUTO_TIMING_UPDATE
 * streams only. */
void pa_stream_set_latency_update_callback(pa_stream *p, pa_stream_notify_cb_t cb, void *userdata);

/** Set the callback function that is called whenever the stream is
 * moved to a different sink/source. Use pa_stream_get_device_name() or
 * pa_stream_get_device_index() to query the new sink/source. This
 * notification is only generated when the server is at least
 * 0.9.8. \since 0.9.8 */
void pa_stream_set_moved_callback(pa_stream *p, pa_stream_notify_cb_t cb, void *userdata);

/** Set the callback function that is called whenever the sink/source
 * this stream is connected to is suspended or resumed. Use
 * pa_stream_is_suspended() to query the new suspend status. Please
 * note that the suspend status might also change when the stream is
 * moved between devices. Thus if you call this function you very
 * likely want to call pa_stream_set_moved_callback() too. This
 * notification is only generated when the server is at least
 * 0.9.8. \since 0.9.8 */
void pa_stream_set_suspended_callback(pa_stream *p, pa_stream_notify_cb_t cb, void *userdata);

/** Set the callback function that is called whenever a meta/policy
 * control event is received. \since 0.9.15 */
void pa_stream_set_event_callback(pa_stream *p, pa_stream_event_cb_t cb, void *userdata);

/** Set the callback function that is called whenever the buffer
 * attributes on the server side change. Please note that the buffer
 * attributes can change when moving a stream to a different
 * sink/source too, hence if you use this callback you should use
 * pa_stream_set_moved_callback() as well. \since 0.9.15 */
void pa_stream_set_buffer_attr_callback(pa_stream *p, pa_stream_notify_cb_t cb, void *userdata);

/** Pause (or resume) playback of this stream temporarily. Available
 * on both playback and recording streams. If \a b is 1 the stream is
 * paused. If \a b is 0 the stream is resumed. The pause/resume operation
 * is executed as quickly as possible. If a cork is very quickly
 * followed by an uncork or the other way round, this might not
 * actually have any effect on the stream that is output. You can use
 * pa_stream_is_corked() to find out whether the stream is currently
 * paused or not. Normally a stream will be created in uncorked
 * state. If you pass PA_STREAM_START_CORKED as a flag when connecting
 * the stream, it will be created in corked state. */
pa_operation* pa_stream_cork(pa_stream *s, int b, pa_stream_success_cb_t cb, void *userdata);

/** Flush the playback or record buffer of this stream. This discards any audio data
 * in the buffer.  Most of the time you're better off using the parameter
 * \a seek of pa_stream_write() instead of this function. */
pa_operation* pa_stream_flush(pa_stream *s, pa_stream_success_cb_t cb, void *userdata);

/** Reenable prebuffering if specified in the pa_buffer_attr
 * structure. Available for playback streams only. */
pa_operation* pa_stream_prebuf(pa_stream *s, pa_stream_success_cb_t cb, void *userdata);

/** Request immediate start of playback on this stream. This disables
 * prebuffering temporarily if specified in the pa_buffer_attr structure.
 * Available for playback streams only. */
pa_operation* pa_stream_trigger(pa_stream *s, pa_stream_success_cb_t cb, void *userdata);

/** Rename the stream. */
pa_operation* pa_stream_set_name(pa_stream *s, const char *name, pa_stream_success_cb_t cb, void *userdata);

/** Return the current playback/recording time. This is based on the
 * data in the timing info structure returned by
 * pa_stream_get_timing_info().
 *
 * This function will usually only return new data if a timing info
 * update has been received. Only if timing interpolation has been
 * requested (PA_STREAM_INTERPOLATE_TIMING) the data from the last
 * timing update is used for an estimation of the current
 * playback/recording time based on the local time that passed since
 * the timing info structure has been acquired.
 *
 * The time value returned by this function is guaranteed to increase
 * monotonically (the returned value is always greater
 * or equal to the value returned by the last call). This behaviour
 * can be disabled by using PA_STREAM_NOT_MONOTONIC. This may be
 * desirable to better deal with bad estimations of transport
 * latencies, but may have strange effects if the application is not
 * able to deal with time going 'backwards'.
 *
 * The time interpolator activated by PA_STREAM_INTERPOLATE_TIMING
 * favours 'smooth' time graphs over accurate ones to improve the
 * smoothness of UI operations that are tied to the audio clock. If
 * accuracy is more important to you, you might need to estimate your
 * timing based on the data from pa_stream_get_timing_info() yourself
 * or not work with interpolated timing at all and instead always
 * query the server side for the most up to date timing with
 * pa_stream_update_timing_info().
 *
 * If no timing information has been
 * received yet this call will return -PA_ERR_NODATA. For more details
 * see pa_stream_get_timing_info(). */
int pa_stream_get_time(pa_stream *s, pa_usec_t *r_usec);

/** Determine the total stream latency. This function is based on
 * pa_stream_get_time().
 *
 * The latency is stored in \a *r_usec. In case the stream is a
 * monitoring stream the result can be negative, i.e. the captured
 * samples are not yet played. In this case \a *negative is set to 1.
 *
 * If no timing information has been received yet, this call will
 * return -PA_ERR_NODATA. On success, it will return 0.
 *
 * For more details see pa_stream_get_timing_info() and
 * pa_stream_get_time(). */
int pa_stream_get_latency(pa_stream *s, pa_usec_t *r_usec, int *negative);

/** Return the latest raw timing data structure. The returned pointer
 * refers to an internal read-only instance of the timing
 * structure. The user should make a copy of this structure if he
 * wants to modify it. An in-place update to this data structure may
 * be requested using pa_stream_update_timing_info().
 *
 * If no timing information has been received before (i.e. by
 * requesting pa_stream_update_timing_info() or by using
 * PA_STREAM_AUTO_TIMING_UPDATE), this function will fail with
 * -PA_ERR_NODATA.
 *
 * Please note that the write_index member field (and only this field)
 * is updated on each pa_stream_write() call, not just when a timing
 * update has been received. */
const pa_timing_info* pa_stream_get_timing_info(pa_stream *s);

/** Return a pointer to the stream's sample specification. */
const pa_sample_spec* pa_stream_get_sample_spec(pa_stream *s);

/** Return a pointer to the stream's channel map. */
const pa_channel_map* pa_stream_get_channel_map(pa_stream *s);

/** Return a pointer to the stream's format. \since 1.0 */
const pa_format_info* pa_stream_get_format_info(pa_stream *s);

/** Return the per-stream server-side buffer metrics of the
 * stream. Only valid after the stream has been connected successfully
 * and if the server is at least PulseAudio 0.9. This will return the
 * actual configured buffering metrics, which may differ from what was
 * requested during pa_stream_connect_record() or
 * pa_stream_connect_playback(). This call will always return the
 * actual per-stream server-side buffer metrics, regardless whether
 * PA_STREAM_ADJUST_LATENCY is set or not. \since 0.9.0 */
const pa_buffer_attr* pa_stream_get_buffer_attr(pa_stream *s);

/** Change the buffer metrics of the stream during playback. The
 * server might have chosen different buffer metrics then
 * requested. The selected metrics may be queried with
 * pa_stream_get_buffer_attr() as soon as the callback is called. Only
 * valid after the stream has been connected successfully and if the
 * server is at least PulseAudio 0.9.8. Please be aware of the
 * slightly different semantics of the call depending whether
 * PA_STREAM_ADJUST_LATENCY is set or not. \since 0.9.8 */
pa_operation *pa_stream_set_buffer_attr(pa_stream *s, const pa_buffer_attr *attr, pa_stream_success_cb_t cb, void *userdata);

/** Change the stream sampling rate during playback. You need to pass
 * PA_STREAM_VARIABLE_RATE in the flags parameter of
 * pa_stream_connect_playback() if you plan to use this function. Only valid
 * after the stream has been connected successfully and if the server
 * is at least PulseAudio 0.9.8. \since 0.9.8 */
pa_operation *pa_stream_update_sample_rate(pa_stream *s, uint32_t rate, pa_stream_success_cb_t cb, void *userdata);

/** Update the property list of the sink input/source output of this
 * stream, adding new entries. Please note that it is highly
 * recommended to set as many properties initially via
 * pa_stream_new_with_proplist() as possible instead a posteriori with
 * this function, since that information may be used to route
 * this stream to the right device. \since 0.9.11 */
pa_operation *pa_stream_proplist_update(pa_stream *s, pa_update_mode_t mode, pa_proplist *p, pa_stream_success_cb_t cb, void *userdata);

/** Update the property list of the sink input/source output of this
 * stream, remove entries. \since 0.9.11 */
pa_operation *pa_stream_proplist_remove(pa_stream *s, const char *const keys[], pa_stream_success_cb_t cb, void *userdata);

/** For record streams connected to a monitor source: monitor only a
 * very specific sink input of the sink. This function needs to be
 * called before pa_stream_connect_record() is called. \since
 * 0.9.11 */
int pa_stream_set_monitor_stream(pa_stream *s, uint32_t sink_input_idx);

/** Return the sink input index previously set with
 * pa_stream_set_monitor_stream().
 * \since 0.9.11 */
uint32_t pa_stream_get_monitor_stream(pa_stream *s);

PA_C_DECL_END

#endif
