#ifndef foothreadmainloophfoo
#define foothreadmainloophfoo

/***
  This file is part of PulseAudio.

  Copyright 2006 Lennart Poettering
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

#include <pulse/mainloop-api.h>
#include <pulse/cdecl.h>
#include <pulse/version.h>

PA_C_DECL_BEGIN

/** \page threaded_mainloop Threaded Main Loop
 *
 * \section overv_sec Overview
 *
 * The threaded main loop implementation is a special version of the primary
 * main loop implementation (see \ref mainloop). For the basic design, see
 * its documentation.
 *
 * The added feature in the threaded main loop is that it spawns a new thread
 * that runs the real main loop. This allows a synchronous application to use
 * the asynchronous API without risking to stall the PulseAudio library.
 *
 * \section creat_sec Creation
 *
 * A pa_threaded_mainloop object is created using pa_threaded_mainloop_new().
 * This will only allocate the required structures though, so to use it the
 * thread must also be started. This is done through
 * pa_threaded_mainloop_start(), after which you can start using the main loop.
 *
 * \section destr_sec Destruction
 *
 * When the PulseAudio connection has been terminated, the thread must be
 * stopped and the resources freed. Stopping the thread is done using
 * pa_threaded_mainloop_stop(), which must be called without the lock (see
 * below) held. When that function returns, the thread is stopped and the
 * pa_threaded_mainloop object can be freed using pa_threaded_mainloop_free().
 *
 * \section lock_sec Locking
 *
 * Since the PulseAudio API doesn't allow concurrent accesses to objects,
 * a locking scheme must be used to guarantee safe usage. The threaded main
 * loop API provides such a scheme through the functions
 * pa_threaded_mainloop_lock() and pa_threaded_mainloop_unlock().
 *
 * The lock is recursive, so it's safe to use it multiple times from the same
 * thread. Just make sure you call pa_threaded_mainloop_unlock() the same
 * number of times you called pa_threaded_mainloop_lock().
 *
 * The lock needs to be held whenever you call any PulseAudio function that
 * uses an object associated with this main loop. Make sure you do not hold
 * on to the lock more than necessary though, as the threaded main loop stops
 * while the lock is held.
 *
 * Example:
 *
 * \code
 * void my_check_stream_func(pa_threaded_mainloop *m, pa_stream *s) {
 *     pa_stream_state_t state;
 *
 *     pa_threaded_mainloop_lock(m);
 *
 *     state = pa_stream_get_state(s);
 *
 *     pa_threaded_mainloop_unlock(m);
 *
 *     if (state == PA_STREAM_READY)
 *         printf("Stream is ready!");
 *     else
 *         printf("Stream is not ready!");
 * }
 * \endcode
 *
 * \section cb_sec Callbacks
 *
 * Callbacks in PulseAudio are asynchronous, so they require extra care when
 * using them together with a threaded main loop.
 *
 * The easiest way to turn the callback based operations into synchronous
 * ones, is to simply wait for the callback to be called and continue from
 * there. This is the approach chosen in PulseAudio's threaded API.
 *
 * \subsection basic_subsec Basic callbacks
 *
 * For the basic case, where all that is required is to wait for the callback
 * to be invoked, the code should look something like this:
 *
 * Example:
 *
 * \code
 * static void my_drain_callback(pa_stream *s, int success, void *userdata) {
 *     pa_threaded_mainloop *m;
 *
 *     m = userdata;
 *     assert(m);
 *
 *     pa_threaded_mainloop_signal(m, 0);
 * }
 *
 * void my_drain_stream_func(pa_threaded_mainloop *m, pa_stream *s) {
 *     pa_operation *o;
 *
 *     pa_threaded_mainloop_lock(m);
 *
 *     o = pa_stream_drain(s, my_drain_callback, m);
 *     assert(o);
 *
 *     while (pa_operation_get_state(o) == PA_OPERATION_RUNNING)
 *         pa_threaded_mainloop_wait(m);
 *
 *     pa_operation_unref(o);
 *
 *     pa_threaded_mainloop_unlock(m);
 * }
 * \endcode
 *
 * The main function, my_drain_stream_func(), will wait for the callback to
 * be called using pa_threaded_mainloop_wait().
 *
 * If your application is multi-threaded, then this waiting must be
 * done inside a while loop. The reason for this is that multiple
 * threads might be using pa_threaded_mainloop_wait() at the same
 * time. Each thread must therefore verify that it was its callback
 * that was invoked. Also the underlying OS synchronization primitives
 * are usually not free of spurious wake-ups, so a
 * pa_threaded_mainloop_wait() must be called within a loop even if
 * you have only one thread waiting.
 *
 * The callback, my_drain_callback(), indicates to the main function that it
 * has been called using pa_threaded_mainloop_signal().
 *
 * As you can see, pa_threaded_mainloop_wait() may only be called with
 * the lock held. The same thing is true for pa_threaded_mainloop_signal(),
 * but as the lock is held before the callback is invoked, you do not have to
 * deal with that.
 *
 * The functions will not dead lock because the wait function will release
 * the lock before waiting and then regrab it once it has been signalled.
 * For those of you familiar with threads, the behaviour is that of a
 * condition variable.
 *
 * \subsection data_subsec Data callbacks
 *
 * For many callbacks, simply knowing that they have been called is
 * insufficient. The callback also receives some data that is desired. To
 * access this data safely, we must extend our example a bit:
 *
 * \code
 * static int * volatile drain_result = NULL;
 *
 * static void my_drain_callback(pa_stream*s, int success, void *userdata) {
 *     pa_threaded_mainloop *m;
 *
 *     m = userdata;
 *     assert(m);
 *
 *     drain_result = &success;
 *
 *     pa_threaded_mainloop_signal(m, 1);
 * }
 *
 * void my_drain_stream_func(pa_threaded_mainloop *m, pa_stream *s) {
 *     pa_operation *o;
 *
 *     pa_threaded_mainloop_lock(m);
 *
 *     o = pa_stream_drain(s, my_drain_callback, m);
 *     assert(o);
 *
 *     while (drain_result == NULL)
 *         pa_threaded_mainloop_wait(m);
 *
 *     pa_operation_unref(o);
 *
 *     if (*drain_result)
 *         printf("Success!");
 *     else
 *         printf("Bitter defeat...");
 *
 *     pa_threaded_mainloop_accept(m);
 *
 *     pa_threaded_mainloop_unlock(m);
 * }
 * \endcode
 *
 * The example is a bit silly as it would probably have been easier to just
 * copy the contents of success, but for larger data structures this can be
 * wasteful.
 *
 * The difference here compared to the basic callback is the value 1 passed
 * to pa_threaded_mainloop_signal() and the call to
 * pa_threaded_mainloop_accept(). What will happen is that
 * pa_threaded_mainloop_signal() will signal the main function and then wait.
 * The main function is then free to use the data in the callback until
 * pa_threaded_mainloop_accept() is called, which will allow the callback
 * to continue.
 *
 * Note that pa_threaded_mainloop_accept() must be called some time between
 * exiting the while loop and unlocking the main loop! Failure to do so will
 * result in a race condition. I.e. it is not ok to release the lock and
 * regrab it before calling pa_threaded_mainloop_accept().
 *
 * \subsection async_subsec Asynchronous callbacks
 *
 * PulseAudio also has callbacks that are completely asynchronous, meaning
 * that they can be called at any time. The threaded main loop API provides
 * the locking mechanism to handle concurrent accesses, but nothing else.
 * Applications will have to handle communication from the callback to the
 * main program through their own mechanisms.
 *
 * The callbacks that are completely asynchronous are:
 *
 * \li State callbacks for contexts, streams, etc.
 * \li Subscription notifications
 */

/** \file
 *
 * A thread based event loop implementation based on pa_mainloop. The
 * event loop is run in a helper thread in the background. A few
 * synchronization primitives are available to access the objects
 * attached to the event loop safely.
 *
 * See also \subpage threaded_mainloop
 */

/** An opaque threaded main loop object */
typedef struct pa_threaded_mainloop pa_threaded_mainloop;

/** Allocate a new threaded main loop object. You have to call
 * pa_threaded_mainloop_start() before the event loop thread starts
 * running. */
pa_threaded_mainloop *pa_threaded_mainloop_new(void);

/** Free a threaded main loop object. If the event loop thread is
 * still running, terminate it with pa_threaded_mainloop_stop()
 * first. */
void pa_threaded_mainloop_free(pa_threaded_mainloop* m);

/** Start the event loop thread. */
int pa_threaded_mainloop_start(pa_threaded_mainloop *m);

/** Terminate the event loop thread cleanly. Make sure to unlock the
 * mainloop object before calling this function. */
void pa_threaded_mainloop_stop(pa_threaded_mainloop *m);

/** Lock the event loop object, effectively blocking the event loop
 * thread from processing events. You can use this to enforce
 * exclusive access to all objects attached to the event loop. This
 * lock is recursive. This function may not be called inside the event
 * loop thread. Events that are dispatched from the event loop thread
 * are executed with this lock held. */
void pa_threaded_mainloop_lock(pa_threaded_mainloop *m);

/** Unlock the event loop object, inverse of pa_threaded_mainloop_lock(). */
void pa_threaded_mainloop_unlock(pa_threaded_mainloop *m);

/** Wait for an event to be signalled by the event loop thread. You
 * can use this to pass data from the event loop thread to the main
 * thread in a synchronized fashion. This function may not be called
 * inside the event loop thread. Prior to this call the event loop
 * object needs to be locked using pa_threaded_mainloop_lock(). While
 * waiting the lock will be released. Immediately before returning it
 * will be acquired again. This function may spuriously wake up even
 * without pa_threaded_mainloop_signal() being called. You need to
 * make sure to handle that! */
void pa_threaded_mainloop_wait(pa_threaded_mainloop *m);

/** Signal all threads waiting for a signalling event in
 * pa_threaded_mainloop_wait(). If wait_for_accept is non-zero, do
 * not return before the signal was accepted by a
 * pa_threaded_mainloop_accept() call. While waiting for that condition
 * the event loop object is unlocked. */
void pa_threaded_mainloop_signal(pa_threaded_mainloop *m, int wait_for_accept);

/** Accept a signal from the event thread issued with
 * pa_threaded_mainloop_signal(). This call should only be used in
 * conjunction with pa_threaded_mainloop_signal() with a non-zero
 * wait_for_accept value.  */
void pa_threaded_mainloop_accept(pa_threaded_mainloop *m);

/** Return the return value as specified with the main loop's
 * pa_mainloop_quit() routine. */
int pa_threaded_mainloop_get_retval(pa_threaded_mainloop *m);

/** Return the main loop abstraction layer vtable for this main loop.
 * There is no need to free this object as it is owned by the loop
 * and is destroyed when the loop is freed. */
pa_mainloop_api* pa_threaded_mainloop_get_api(pa_threaded_mainloop*m);

/** Returns non-zero when called from within the event loop thread. \since 0.9.7 */
int pa_threaded_mainloop_in_thread(pa_threaded_mainloop *m);

/** Sets the name of the thread. \since 5.0 */
void pa_threaded_mainloop_set_name(pa_threaded_mainloop *m, const char *name);

PA_C_DECL_END

#endif
