#ifndef foooperationhfoo
#define foooperationhfoo

/***
  This file is part of PulseAudio.

  Copyright 2004-2006 Lennart Poettering

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

#include <pulse/cdecl.h>
#include <pulse/def.h>
#include <pulse/version.h>

/** \file
 * Asynchronous operations */

PA_C_DECL_BEGIN

/** An asynchronous operation object */
typedef struct pa_operation pa_operation;

/** A callback for operation state changes */
typedef void (*pa_operation_notify_cb_t) (pa_operation *o, void *userdata);

/** Increase the reference count by one */
pa_operation *pa_operation_ref(pa_operation *o);

/** Decrease the reference count by one */
void pa_operation_unref(pa_operation *o);

/** Cancel the operation. Beware! This will not necessarily cancel the
 * execution of the operation on the server side. However it will make
 * sure that the callback associated with this operation will not be
 * called anymore, effectively disabling the operation from the client
 * side's view. */
void pa_operation_cancel(pa_operation *o);

/** Return the current status of the operation */
pa_operation_state_t pa_operation_get_state(pa_operation *o);

/** Set the callback function that is called when the operation state
 * changes. Usually this is not necessary, since the functions that
 * create pa_operation objects already take a callback that is called
 * when the operation finishes. Registering a state change callback is
 * mainly useful, if you want to get called back also if the operation
 * gets cancelled. \since 4.0 */
void pa_operation_set_state_callback(pa_operation *o, pa_operation_notify_cb_t cb, void *userdata);

PA_C_DECL_END

#endif
