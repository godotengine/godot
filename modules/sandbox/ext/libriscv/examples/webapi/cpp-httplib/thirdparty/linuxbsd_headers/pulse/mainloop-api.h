#ifndef foomainloopapihfoo
#define foomainloopapihfoo

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

#include <sys/time.h>

#include <pulse/cdecl.h>
#include <pulse/version.h>

/** \file
 *
 * Main loop abstraction layer. Both the PulseAudio core and the
 * PulseAudio client library use a main loop abstraction layer. Due to
 * this it is possible to embed PulseAudio into other
 * applications easily. Two main loop implementations are
 * currently available:
 * \li A minimal implementation based on the C library's poll() function (See \ref mainloop.h)
 * \li A wrapper around the GLIB main loop. Use this to embed PulseAudio into your GLIB/GTK+/GNOME programs (See \ref glib-mainloop.h)
 *
 * The structure pa_mainloop_api is used as vtable for the main loop abstraction.
 *
 * This mainloop abstraction layer has no direct support for UNIX signals. Generic, mainloop implementation agnostic support is available through \ref mainloop-signal.h.
 * */

PA_C_DECL_BEGIN

/** An abstract mainloop API vtable */
typedef struct pa_mainloop_api pa_mainloop_api;

/** A bitmask for IO events */
typedef enum pa_io_event_flags {
    PA_IO_EVENT_NULL = 0,     /**< No event */
    PA_IO_EVENT_INPUT = 1,    /**< Input event */
    PA_IO_EVENT_OUTPUT = 2,   /**< Output event */
    PA_IO_EVENT_HANGUP = 4,   /**< Hangup event */
    PA_IO_EVENT_ERROR = 8     /**< Error event */
} pa_io_event_flags_t;

/** An opaque IO event source object */
typedef struct pa_io_event pa_io_event;
/** An IO event callback prototype \since 0.9.3 */
typedef void (*pa_io_event_cb_t)(pa_mainloop_api*ea, pa_io_event* e, int fd, pa_io_event_flags_t events, void *userdata);
/** A IO event destroy callback prototype \since 0.9.3 */
typedef void (*pa_io_event_destroy_cb_t)(pa_mainloop_api*a, pa_io_event *e, void *userdata);

/** An opaque timer event source object */
typedef struct pa_time_event pa_time_event;
/** A time event callback prototype \since 0.9.3 */
typedef void (*pa_time_event_cb_t)(pa_mainloop_api*a, pa_time_event* e, const struct timeval *tv, void *userdata);
/** A time event destroy callback prototype \since 0.9.3 */
typedef void (*pa_time_event_destroy_cb_t)(pa_mainloop_api*a, pa_time_event *e, void *userdata);

/** An opaque deferred event source object. Events of this type are triggered once in every main loop iteration */
typedef struct pa_defer_event pa_defer_event;
/** A defer event callback prototype \since 0.9.3 */
typedef void (*pa_defer_event_cb_t)(pa_mainloop_api*a, pa_defer_event* e, void *userdata);
/** A defer event destroy callback prototype \since 0.9.3 */
typedef void (*pa_defer_event_destroy_cb_t)(pa_mainloop_api*a, pa_defer_event *e, void *userdata);

/** An abstract mainloop API vtable */
struct pa_mainloop_api {
    /** A pointer to some private, arbitrary data of the main loop implementation */
    void *userdata;

    /** Create a new IO event source object */
    pa_io_event* (*io_new)(pa_mainloop_api*a, int fd, pa_io_event_flags_t events, pa_io_event_cb_t cb, void *userdata);
    /** Enable or disable IO events on this object */
    void (*io_enable)(pa_io_event* e, pa_io_event_flags_t events);
    /** Free a IO event source object */
    void (*io_free)(pa_io_event* e);
    /** Set a function that is called when the IO event source is destroyed. Use this to free the userdata argument if required */
    void (*io_set_destroy)(pa_io_event *e, pa_io_event_destroy_cb_t cb);

    /** Create a new timer event source object for the specified Unix time */
    pa_time_event* (*time_new)(pa_mainloop_api*a, const struct timeval *tv, pa_time_event_cb_t cb, void *userdata);
    /** Restart a running or expired timer event source with a new Unix time */
    void (*time_restart)(pa_time_event* e, const struct timeval *tv);
    /** Free a deferred timer event source object */
    void (*time_free)(pa_time_event* e);
    /** Set a function that is called when the timer event source is destroyed. Use this to free the userdata argument if required */
    void (*time_set_destroy)(pa_time_event *e, pa_time_event_destroy_cb_t cb);

    /** Create a new deferred event source object */
    pa_defer_event* (*defer_new)(pa_mainloop_api*a, pa_defer_event_cb_t cb, void *userdata);
    /** Enable or disable a deferred event source temporarily */
    void (*defer_enable)(pa_defer_event* e, int b);
    /** Free a deferred event source object */
    void (*defer_free)(pa_defer_event* e);
    /** Set a function that is called when the deferred event source is destroyed. Use this to free the userdata argument if required */
    void (*defer_set_destroy)(pa_defer_event *e, pa_defer_event_destroy_cb_t cb);

    /** Exit the main loop and return the specified retval*/
    void (*quit)(pa_mainloop_api*a, int retval);
};

/** Run the specified callback function once from the main loop using an
 * anonymous defer event. If the mainloop runs in a different thread, you need
 * to follow the mainloop implementation's rules regarding how to safely create
 * defer events. In particular, if you're using \ref pa_threaded_mainloop, you
 * must lock the mainloop before calling this function. */
void pa_mainloop_api_once(pa_mainloop_api*m, void (*callback)(pa_mainloop_api*m, void *userdata), void *userdata);

PA_C_DECL_END

#endif
