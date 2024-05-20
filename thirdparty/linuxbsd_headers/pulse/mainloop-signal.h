#ifndef foomainloopsignalhfoo
#define foomainloopsignalhfoo

/***
  This file is part of PulseAudio.

  Copyright 2004-2008 Lennart Poettering
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

/** \file
 * UNIX signal support for main loops. In contrast to other
 * main loop event sources such as timer and IO events, UNIX signal
 * support requires modification of the global process
 * environment. Due to this the generic main loop abstraction layer as
 * defined in \ref mainloop-api.h doesn't have direct support for UNIX
 * signals. However, you may hook signal support into an abstract main loop via the routines defined herein.
 */

/** An opaque UNIX signal event source object */
typedef struct pa_signal_event pa_signal_event;

/** Callback prototype for signal events */
typedef void (*pa_signal_cb_t) (pa_mainloop_api *api, pa_signal_event*e, int sig, void *userdata);

/** Destroy callback prototype for signal events */
typedef void (*pa_signal_destroy_cb_t) (pa_mainloop_api *api, pa_signal_event*e, void *userdata);

/** Initialize the UNIX signal subsystem and bind it to the specified main loop */
int pa_signal_init(pa_mainloop_api *api);

/** Cleanup the signal subsystem */
void pa_signal_done(void);

/** Create a new UNIX signal event source object */
pa_signal_event* pa_signal_new(int sig, pa_signal_cb_t callback, void *userdata);

/** Free a UNIX signal event source object */
void pa_signal_free(pa_signal_event *e);

/** Set a function that is called when the signal event source is destroyed. Use this to free the userdata argument if required */
void pa_signal_set_destroy(pa_signal_event *e, pa_signal_destroy_cb_t callback);

PA_C_DECL_END

#endif
