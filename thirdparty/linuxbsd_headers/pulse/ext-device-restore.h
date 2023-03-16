#ifndef foopulseextdevicerestorehfoo
#define foopulseextdevicerestorehfoo

/***
  This file is part of PulseAudio.

  Copyright 2008 Lennart Poettering
  Copyright 2011 Colin Guthrie

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

#include <pulse/context.h>
#include <pulse/format.h>
#include <pulse/version.h>

/** \file
 *
 * Routines for controlling module-device-restore
 */

PA_C_DECL_BEGIN

/** Stores information about one device in the device database that is
 * maintained by module-device-manager. \since 1.0 */
typedef struct pa_ext_device_restore_info {
    pa_device_type_t type;       /**< Device type sink or source? */
    uint32_t index;              /**< The device index */
    uint8_t n_formats;           /**< How many formats do we have? */
    pa_format_info **formats;    /**< An array of formats (may be NULL if n_formats == 0) */
} pa_ext_device_restore_info;

/** Callback prototype for pa_ext_device_restore_test(). \since 1.0 */
typedef void (*pa_ext_device_restore_test_cb_t)(
        pa_context *c,
        uint32_t version,
        void *userdata);

/** Test if this extension module is available in the server. \since 1.0 */
pa_operation *pa_ext_device_restore_test(
        pa_context *c,
        pa_ext_device_restore_test_cb_t cb,
        void *userdata);

/** Subscribe to changes in the device database. \since 1.0 */
pa_operation *pa_ext_device_restore_subscribe(
        pa_context *c,
        int enable,
        pa_context_success_cb_t cb,
        void *userdata);

/** Callback prototype for pa_ext_device_restore_set_subscribe_cb(). \since 1.0 */
typedef void (*pa_ext_device_restore_subscribe_cb_t)(
        pa_context *c,
        pa_device_type_t type,
        uint32_t idx,
        void *userdata);

/** Set the subscription callback that is called when
 * pa_ext_device_restore_subscribe() was called. \since 1.0 */
void pa_ext_device_restore_set_subscribe_cb(
        pa_context *c,
        pa_ext_device_restore_subscribe_cb_t cb,
        void *userdata);

/** Callback prototype for pa_ext_device_restore_read_formats(). \since 1.0 */
typedef void (*pa_ext_device_restore_read_device_formats_cb_t)(
        pa_context *c,
        const pa_ext_device_restore_info *info,
        int eol,
        void *userdata);

/** Read the formats for all present devices from the device database. \since 1.0 */
pa_operation *pa_ext_device_restore_read_formats_all(
        pa_context *c,
        pa_ext_device_restore_read_device_formats_cb_t cb,
        void *userdata);

/** Read an entry from the device database. \since 1.0 */
pa_operation *pa_ext_device_restore_read_formats(
        pa_context *c,
        pa_device_type_t type,
        uint32_t idx,
        pa_ext_device_restore_read_device_formats_cb_t cb,
        void *userdata);

/** Read an entry from the device database. \since 1.0 */
pa_operation *pa_ext_device_restore_save_formats(
        pa_context *c,
        pa_device_type_t type,
        uint32_t idx,
        uint8_t n_formats,
        pa_format_info **formats,
        pa_context_success_cb_t cb,
        void *userdata);

PA_C_DECL_END

#endif
