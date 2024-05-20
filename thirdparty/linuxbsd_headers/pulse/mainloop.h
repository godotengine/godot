#ifndef foomainloophfoo
#define foomainloophfoo

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

#include <pulse/mainloop-api.h>
#include <pulse/cdecl.h>

PA_C_DECL_BEGIN

struct pollfd;

/** \page mainloop Main Loop
 *
 * \section overv_sec Overview
 *
 * The built-in main loop implementation is based on the poll() system call.
 * It supports the functions defined in the main loop abstraction and very
 * little else.
 *
 * The main loop is created using pa_mainloop_new() and destroyed using
 * pa_mainloop_free(). To get access to the main loop abstraction,
 * pa_mainloop_get_api() is used.
 *
 * \section iter_sec Iteration
 *
 * The main loop is designed around the concept of iterations. Each iteration
 * consists of three steps that repeat during the application's entire
 * lifetime:
 *
 * -# Prepare - Build a list of file descriptors
 *               that need to be monitored and calculate the next timeout.
 * -# Poll - Execute the actual poll() system call.
 * -# Dispatch - Dispatch any events that have fired.
 *
 * When using the main loop, the application can either execute each
 * iteration, one at a time, using pa_mainloop_iterate(), or let the library
 * iterate automatically using pa_mainloop_run().
 *
 * \section thread_sec Threads
 *
 * The main loop functions are designed to be thread safe, but the objects
 * are not. What this means is that multiple main loops can be used, but only
 * one object per thread.
 *
 */

/** \file
 *
 * A minimal main loop implementation based on the C library's poll()
 * function. Using the routines defined herein you may create a simple
 * main loop supporting the generic main loop abstraction layer as
 * defined in \ref mainloop-api.h. This implementation is thread safe
 * as long as you access the main loop object from a single thread only.
 *
 * See also \subpage mainloop
 */

/** An opaque main loop object */
typedef struct pa_mainloop pa_mainloop;

/** Allocate a new main loop object */
pa_mainloop *pa_mainloop_new(void);

/** Free a main loop object */
void pa_mainloop_free(pa_mainloop* m);

/** Prepare for a single iteration of the main loop. Returns a negative value
on error or exit request. timeout specifies a maximum timeout for the subsequent
poll, or -1 for blocking behaviour. .*/
int pa_mainloop_prepare(pa_mainloop *m, int timeout);

/** Execute the previously prepared poll. Returns a negative value on error.*/
int pa_mainloop_poll(pa_mainloop *m);

/** Dispatch timeout, io and deferred events from the previously executed poll. Returns
a negative value on error. On success returns the number of source dispatched. */
int pa_mainloop_dispatch(pa_mainloop *m);

/** Return the return value as specified with the main loop's quit() routine. */
int pa_mainloop_get_retval(pa_mainloop *m);

/** Run a single iteration of the main loop. This is a convenience function
for pa_mainloop_prepare(), pa_mainloop_poll() and pa_mainloop_dispatch().
Returns a negative value on error or exit request. If block is nonzero,
block for events if none are queued. Optionally return the return value as
specified with the main loop's quit() routine in the integer variable retval points
to. On success returns the number of sources dispatched in this iteration. */
int pa_mainloop_iterate(pa_mainloop *m, int block, int *retval);

/** Run unlimited iterations of the main loop object until the main loop's quit() routine is called. */
int pa_mainloop_run(pa_mainloop *m, int *retval);

/** Return the abstract main loop abstraction layer vtable for this
    main loop. No need to free the API as it is owned by the loop
    and is destroyed when the loop is freed. */
pa_mainloop_api* pa_mainloop_get_api(pa_mainloop*m);

/** Shutdown the main loop with the specified return value */
void pa_mainloop_quit(pa_mainloop *m, int retval);

/** Interrupt a running poll (for threaded systems) */
void pa_mainloop_wakeup(pa_mainloop *m);

/** Generic prototype of a poll() like function */
typedef int (*pa_poll_func)(struct pollfd *ufds, unsigned long nfds, int timeout, void*userdata);

/** Change the poll() implementation */
void pa_mainloop_set_poll_func(pa_mainloop *m, pa_poll_func poll_func, void *userdata);

PA_C_DECL_END

#endif
