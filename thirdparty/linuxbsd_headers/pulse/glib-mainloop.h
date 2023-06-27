#ifndef fooglibmainloophfoo
#define fooglibmainloophfoo

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

#include <glib.h>

#include <pulse/mainloop-api.h>
#include <pulse/cdecl.h>
#include <pulse/version.h>

/** \page glib-mainloop GLIB Main Loop Bindings
 *
 * \section overv_sec Overview
 *
 * The GLIB main loop bindings are extremely easy to use. All that is
 * required is to create a pa_glib_mainloop object using
 * pa_glib_mainloop_new(). When the main loop abstraction is needed, it is
 * provided by pa_glib_mainloop_get_api().
 *
 */

/** \file
 * GLIB main loop support
 *
 * See also \subpage glib-mainloop
 */

PA_C_DECL_BEGIN

/** An opaque GLIB main loop object */
typedef struct pa_glib_mainloop pa_glib_mainloop;

/** Create a new GLIB main loop object for the specified GLIB main
 * loop context. Takes an argument c for the
 * GMainContext to use. If c is NULL the default context is used. */
pa_glib_mainloop *pa_glib_mainloop_new(GMainContext *c);

/** Free the GLIB main loop object */
void pa_glib_mainloop_free(pa_glib_mainloop* g);

/** Return the abstract main loop API vtable for the GLIB main loop
    object. No need to free the API as it is owned by the loop
    and is destroyed when the loop is freed. */
pa_mainloop_api* pa_glib_mainloop_get_api(pa_glib_mainloop *g);

PA_C_DECL_END

#endif
